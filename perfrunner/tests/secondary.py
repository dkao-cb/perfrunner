import time
import urllib2
import base64
import json
import subprocess
import numpy as np
import os
import pdb

from logger import logger

from perfrunner.helpers.cbmonitor import with_stats
from perfrunner.helpers.remote import RemoteHelper
from perfrunner.tests import PerfTest


class SecondaryIndexTest(PerfTest):

    """
    The test measures time it takes to build secondary index. This is just a base
    class, actual measurements happen in initial and incremental secondary indexing tests.
    It benchmarks dumb/bulk indexing.

    Sample test spec snippet:

    [secondary]
    name = myindex1,myindex2
    field = email,city
    index_myindex1_partitions={"email":["5fffff", "7fffff"]}
    index_myindex2_partitions={"city":["5fffff", "7fffff"]}

    NOTE: two partition pivots above imply that we need 3 indexer nodes in the
    cluster spec.
    """

    COLLECTORS = {'secondary_stats': True, 'secondary_debugstats': True}

    def __init__(self, *args):
        super(SecondaryIndexTest, self).__init__(*args)

        """self.test_config.secondaryindex_settings"""
        self.secondaryindex_settings = None
        self.index_nodes = None
        self.index_fields = None
        self.bucket = None
        self.indexes = []
        self.secondaryDB = ''
        self.configfile = ''

        self.secondaryDB = None
        if self.test_config.secondaryindex_settings.db == 'memdb':
            self.secondaryDB = 'memdb'
        logger.info('secondary storage DB..{}'.format(self.secondaryDB))

        self.indexes = self.test_config.secondaryindex_settings.name.split(',')
        self.index_fields = self.test_config.secondaryindex_settings.field.split(",")

        # Get first cluster, its index nodes, and first bucket
        (cluster_name, servers) = \
            self.cluster_spec.yield_servers_by_role('index').next()
        if not servers:
            raise RuntimeError(
                "No index nodes specified for cluster {}".format(cluster_name))
        self.index_nodes = servers

        self.bucket = self.test_config.buckets[0]

        # Generate active index names that are used if there are partitions.
        # Else active_indexes is the same as indexes specified in test config
        self.active_indexes = self.indexes
        num_partitions = None
        for index, field_where in self._get_where_map().iteritems():
            where_list = field_where.itervalues().next()
            num_partitions = len(where_list)
            break
        if num_partitions:
            # overwrite with indexname_0, indexname_1 ... names for each partition
            self.active_indexes = []
            for index in self.indexes:
                for i in xrange(num_partitions):
                    index_i = index + "_{}".format(i)
                    self.active_indexes.append(index_i)
        self.num_partitions = num_partitions

    def _get_where_map(self):
        """
        Given the following in test config:

        [secondary]
        name = myindex1,myindex2
        field = email,city
        index_myindex1_partitions={"email":["5fffff", "afffff"]}
        index_myindex2_partitions={"city":["5fffff", "afffff"]}

        returns something like the following, details omitted by "...":

        {
            "myindex1":
                {"email": [ 'email < "5fffff"', ... ] },
            "myindex2":
                {"city": [ ... , 'city >= "5fffff" and city < "afffff"', city >= "afffff" ]},
        }
        """
        result = {}
        # For each index/field, get the partition pivots in a friendly format.
        # Start with the (index_name, field) pair, find each field's
        # corresponding partition pivots. From the pivots, generate the (low,
        # high) endpoints that define a partition. Use None to represent
        # unbounded.
        for index_name, field in zip(self.indexes, self.index_fields):
            index_partition_name = "index_{}_partitions".format(index_name)
            # check that secondaryindex_settings.index_blah_partitions exists.
            if not hasattr(self.test_config.secondaryindex_settings,
                           index_partition_name):
                continue
            # Value of index_{}_partitions should be a string that resembles a
            # Python dict instead of a JSON due to the support for tuple as
            # keys. However, at the moment the same string can be interpretted
            # as either JSON or Python Dict.
            field_pivots = eval(getattr(self.test_config.secondaryindex_settings,
                                        index_partition_name))
            for field, pivots in field_pivots.iteritems():
                pivots = [None] + pivots + [None]
                partitions = []
                for i in xrange(len(pivots) - 1):
                    partitions.append((pivots[i], pivots[i + 1]))
                if len(partitions) != len(self.index_nodes):
                    raise RuntimeError(
                        "Number of pivots in partitions should be one less" +
                        " than number of index nodes")

            # construct where clause
            where_list = []
            for (left, right) in partitions:
                where = None
                if left and right:
                    where = '\\\"{}\\\" <= {} and {} < \\\"{}\\\"'.format(
                            left, field, field, right)
                elif left:
                    where = '{} >= \\\"{}\\\"'.format(field, left)
                elif right:
                    where = '{} < \\\"{}\\\"'.format(field, right)
                where_list.append(where)

            if index_name not in result:
                result[index_name] = {}
            result[index_name][field] = where_list
        return result

    @with_stats
    def build_secondaryindex(self):
        """call cbindex create command"""
        logger.info('building secondary index..')

        where_map = self._get_where_map()

        self.remote.build_secondary_index(
            self.index_nodes, self.bucket, self.indexes, self.index_fields,
            self.secondaryDB, where_map)

        rest_username, rest_password = self.cluster_spec.rest_credentials
        time_elapsed = self.rest.wait_for_secindex_init_build(self.index_nodes[0].split(':')[0],
                                                              self.active_indexes, rest_username, rest_password)
        return time_elapsed


class InitialSecondaryIndexTest(SecondaryIndexTest):

    """
    The test measures time it takes to build index for the first time. Scenario
    is pretty straightforward, there are only two phases:
    -- Initial data load
    -- Index building
    """

    def build_index(self):
        super(InitialSecondaryIndexTest, self).build_secondaryindex()

    def run(self):
        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        init_ts = time.time()
        self.build_secondaryindex()
        finish_ts = time.time()
        time_elapsed = finish_ts - init_ts
        time_elapsed = self.reporter.finish('Initial secondary index', time_elapsed)
        self.reporter.post_to_sf(
            *self.metric_helper.get_indexing_meta(value=time_elapsed,
                                                  index_type='Initial')
        )


class InitialandIncrementalSecondaryIndexTest(SecondaryIndexTest):

    """
    The test measures time it takes to build index for the first time as well as
    incremental build. There is no disabling of index updates in incremental building,
    index updating is conurrent to KV incremental load.
    """

    def build_initindex(self):
        self.build_secondaryindex()

    @with_stats
    def build_incrindex(self):
        access_settings = self.test_config.access_settings
        self.worker_manager.run_workload(access_settings, self.target_iterator)
        self.worker_manager.wait_for_workers()
        load_settings = self.test_config.load_settings
        access_settings = self.test_config.access_settings
        numitems = load_settings.items + access_settings.items

        self.rest.wait_for_secindex_incr_build(self.index_nodes, self.bucket,
                                               self.active_indexes, numitems)

    def run(self):
        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        from_ts, to_ts = self.build_secondaryindex()
        time_elapsed = (to_ts - from_ts) / 1000.0
        time_elapsed = self.reporter.finish('Initial secondary index', time_elapsed)
        self.reporter.post_to_sf(
            *self.metric_helper.get_indexing_meta(value=time_elapsed,
                                                  index_type='Initial')
        )
        from_ts, to_ts = self.build_incrindex()
        time_elapsed = (to_ts - from_ts) / 1000.0
        time_elapsed = self.reporter.finish('Incremental secondary index', time_elapsed)
        self.reporter.post_to_sf(
            *self.metric_helper.get_indexing_meta(value=time_elapsed,
                                                  index_type='Incremental')
        )


class InitialandIncrementalSecondaryIndexRebalanceTest(InitialandIncrementalSecondaryIndexTest):

    def rebalance(self, initial_nodes, nodes_after):
        clusters = self.cluster_spec.yield_clusters()
        for _, servers in clusters:
            master = servers[0]
            new_nodes = []
            ejected_nodes = []
            new_nodes = enumerate(
                servers[initial_nodes:nodes_after],
                start=initial_nodes
            )
            known_nodes = servers[:nodes_after]
            for i, host_port in new_nodes:
                self.rest.add_node(master, host_port)
        self.rest.rebalance(master, known_nodes, ejected_nodes)

    def run(self):
        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        initial_nodes = []
        nodes_after = [0]
        initial_nodes = self.test_config.cluster.initial_nodes
        nodes_after[0] = initial_nodes[0] + 1
        self.rebalance(initial_nodes[0], nodes_after[0])
        from_ts, to_ts = self.build_secondaryindex()
        time_elapsed = (to_ts - from_ts) / 1000.0
        time_elapsed = self.reporter.finish('Initial secondary index', time_elapsed)
        self.reporter.post_to_sf(
            *self.metric_helper.get_indexing_meta(value=time_elapsed,
                                                  index_type='Initial')
        )
        master = []
        for _, servers in self.cluster_spec.yield_clusters():
            master = servers[0]
        self.monitor.monitor_rebalance(master)
        initial_nodes[0] = initial_nodes[0] + 1
        nodes_after[0] = nodes_after[0] + 1
        self.rebalance(initial_nodes[0], nodes_after[0])
        from_ts, to_ts = self.build_incrindex()
        time_elapsed = (to_ts - from_ts) / 1000.0
        time_elapsed = self.reporter.finish('Incremental secondary index', time_elapsed)
        self.reporter.post_to_sf(
            *self.metric_helper.get_indexing_meta(value=time_elapsed,
                                                  index_type='Incremental')
        )


class _ScanWorkload(object):

    def get_specs(self, concurrency, clients, string_extras={},
                  non_string_extras={}):
        """
        Returns a dictionary that can eventually become a json config file for
        cbindexperf.

        latency tests should pass in {"NInterval" : 100} for both string and
        non-string type of specs.

        Consistency tests should have {"Consistency" : True} in every scan spec
        too.
        """
        result = {}
        result["Concurrency"] = concurrency
        result["Clients"] = clients

        # used later to match the length of field with indexes
        multiplier = 1
        if self.num_partitions:
            multiplier = self.num_partitions

        # "string" type of fields gets a range scan.
        field_types = []
        for index_field in self.index_fields:
            field_type = "non-string"
            if index_field in ('name', 'email', 'alt_email', 'city', 'realm',
                               'country', 'county', 'street'):
                field_type = "string"
            # a list of lists of fields belong to related indexes
            # When one index is partitioned into multiple indexes, these _0, _1
            # ... etc active indexes are "related"
            field_types.append([field_type] * multiplier)

        if len(self.active_indexes) != len(field_types) * multiplier:
            raise RuntimeError(
                "number of indexes do not match with "
                "the number of field types")
        related_indexes = [self.active_indexes[x:x + multiplier] for x in
                           xrange(0, len(self.active_indexes), multiplier)]

        # Find the left boundaries of the scan range
        # boundaries may eventually look like
        # [["0", "5fffff","afffff"],["0", "5fffff","afffff"]]
        boundaries = []
        for index_name, field in zip(self.indexes, self.index_fields):
            # starting bound is always zero
            bounds = ["0"]
            boundaries.append(bounds)
            index_partition_name = "index_{}_partitions".format(index_name)
            # check that secondaryindex_settings.index_blah_partitions exists.
            if not hasattr(self.test_config.secondaryindex_settings,
                           index_partition_name):
                continue
            field_pivots = eval(
                getattr(self.test_config.secondaryindex_settings,
                index_partition_name))
            pivots = field_pivots[field]
            # One can still modify bounds after it is added to boundaries
            bounds.extend(pivots)

        # All information now available to generate scan specs
        i = 0
        scan_specs = []
        for indexes, fields, bounds in zip(
                related_indexes, field_types, boundaries):
            for (index_name, field_type, bound) in zip(indexes, fields, bounds):
                i += 1
                if field_type == "string":
                    left = int(bound, 16)
                    # original low high were 0x15 and 0x28 which yields 21 and
                    # 19 (= 0x28 - 0x15) in decimal.
                    low  = "{:06x}".format(left + 21)
                    high = "{:06x}".format(left + 21 + 19)
                    spec = {
                        "Type": "Range",
                        "Limit": 1,
                        "Repeat": 999999,
                        "Bucket": "bucket-1",
                        "Low": [low],
                        "High": [high],
                        "Id": i,
                        "Inclusion": 3,
                        "Index": index_name,
                        "NInterval": 100
                    }
                    spec.update(string_extras)
                else:
                    spec = {
                        "Type": "All",
                        "Limit": 1,
                        "Repeat": 99,
                        "Bucket": "bucket-1",
                        "Id": i,
                        "Index": index_name,
                        "NInterval": 100
                    }
                scan_specs.append(spec)
        result["ScanSpecs"] = scan_specs
        return result

    def save_specs(self, specs,
                   filename=time.strftime("/tmp/scan_%Y%m%d-%H%M%S.json"),
                   suffix=""):
        filename = filename + suffix
        with open(filename, "w") as f:
            json.dump(specs, f, indent=1, sort_keys=True)
        return filename

    def delete_spec(self, filename):
        os.remove(filename)

    @with_stats
    def apply_scanworkload(self, configfiles):
        rest_username, rest_password = self.cluster_spec.rest_credentials
        logger.info('Initiating scan workload')

        procs = []
        if len(configfiles) > 1:
            raise RuntimeError("This implementation does not support multiple cbindexperfs")
        for i, (config, result) in enumerate(configfiles):
            # wrapped in a pair of unneeded parenthesis to silence flake8
            cmdstr = ((
                "ulimit -n 40960 ; ulimit -a ; cbindexperf{} -cluster {}"
                " -auth=\"{}:{}\" -configfile {}" " -resultfile {} --statsfile"
                " /root/statsfile").format(
                      i,
                      self.index_nodes[0], rest_username, rest_password,
                      config, result))
            logger.info("Running {}".format(cmdstr))
            proc = subprocess.Popen(cmdstr, shell=True)
            procs.append(proc)

        done = [False for p in procs]
        while not all(done):
            for i, proc in enumerate(procs):
                if done[i]:
                    continue
                status = proc.poll()
                if status != None:
                    logger.info("cbindexperf {} status {}".format(i, status))
                    done[i] = True
                    if status != 0:
                        raise Exception('Scan workload could not be applied')
            time.sleep(1)
        logger.info('Scan workload applied')


class SecondaryIndexingThroughputTest(SecondaryIndexTest, _ScanWorkload):

    """
    The test applies scan workload against the 2i server and measures
    and reports the average scan throughput
    """

    COLLECTORS = {'secondary_stats': True, 'secondary_latency': True, 'secondary_debugstats': True}

    def read_scanresults(self):
        sum_scansps = 0
        sum_rowps = 0
        for configfile, result in self.configfiles:
            with open('{}'.format(configfile)) as config_file:
                configdata = json.load(config_file)
                numscans = sum([spec['Repeat'] for spec in configdata['ScanSpecs']])

            logger.info("Reading result.json\n" + open(result).read())
            with open(result) as result_file:
                resdata = json.load(result_file)
            duration_s = (resdata['Duration'])
            numRows = resdata['Rows']
            """scans and rows per sec"""
            scansps = numscans / duration_s
            rowps = numRows / duration_s
            logger.info("single cbindexperf scansps {}".format(scansps))
            sum_scansps += scansps
            sum_rowps += rowps
        return sum_scansps, sum_rowps

    def run(self):
        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        from_ts, to_ts = self.build_secondaryindex()

        # Generate json config file for cbindexperf
        if self.test_config.secondaryindex_settings.stale == 'false':
            specs = self.get_specs(concurrency=1000, clients=1,
                                   string_extras={"Consistency":True},
                                   non_string_extras={"Consistency":True})
        else:
            specs = self.get_specs(concurrency=1000,
                                   clients=1)
        NUM_CBINDEXPERFS = 2
        self.configfiles = []
        for i in xrange(1, NUM_CBINDEXPERFS + 1):
            specs['Id'] = i
            configfile = self.save_specs(specs, suffix="_{}".format(i))
            logger.info("Generated json file:\n" + open(configfile).read())
            self.configfiles.append((configfile, "result{}.json".format(i)))

        self.access_bg()
        self.apply_scanworkload(self.configfiles)
        scanthr, rowthr = self.read_scanresults()
        logger.info('Scan throughput: {}'.format(scanthr))
        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                round(scanthr, 1)
            )
        #self.delete_spec(self.configfile)


class SecondaryIndexingThroughputRebalanceTest(SecondaryIndexingThroughputTest):

    """
    The test applies scan workload against the 2i server and measures
    and reports the average scan throughput"""

    def rebalance(self, initial_nodes, nodes_after):
        clusters = self.cluster_spec.yield_clusters()
        for _, servers in clusters:
            master = servers[0]
            new_nodes = []
            ejected_nodes = []
            new_nodes = enumerate(
                servers[initial_nodes:nodes_after],
                start=initial_nodes
            )
            known_nodes = servers[:nodes_after]
            for i, host_port in new_nodes:
                self.rest.add_node(master, host_port)
        self.rest.rebalance(master, known_nodes, ejected_nodes)

    def run(self):
        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        from_ts, to_ts = self.build_secondaryindex()
        self.access_bg()
        initial_nodes = []
        nodes_after = [0]
        initial_nodes = self.test_config.cluster.initial_nodes
        nodes_after[0] = initial_nodes[0] + 1
        self.rebalance(initial_nodes[0], nodes_after[0])
        self.apply_scanworkload()
        scanthr, rowthr = self.read_scanresults()
        logger.info('Scan throughput: {}'.format(scanthr))
        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                round(scanthr, 1)
            )


class SecondaryIndexingScanLatencyTest(SecondaryIndexTest):

    """
    The test applies scan workload against the 2i server and measures
    and reports the average scan throughput
    """
    COLLECTORS = {'secondary_stats': True, 'secondary_latency': True, 'secondary_debugstats': True}

    @with_stats
    def apply_scanworkload(self):
        rest_username, rest_password = self.cluster_spec.rest_credentials
        logger.info('Initiating scan workload with stats output')
        numindexes = None
        numindexes = len(self.indexes)

        if self.test_config.secondaryindex_settings.stale == 'false':
            if numindexes == 1:
                self.configfile = 'scripts/config_scanlatency_sessionconsistent.json'
            elif numindexes == 5:
                self.configfile = 'scripts/config_scanlatency_sessionconsistent_multiple.json'
        else:
            if numindexes == 1:
                self.configfile = 'scripts/config_scanlatency.json'
            elif numindexes == 5:
                self.configfile = 'scripts/config_scanlatency_multiple.json'

        cmdstr = "cbindexperf -cluster {} -auth=\"{}:{}\" -configfile {} -resultfile result.json -statsfile /root/statsfile".format(self.index_nodes[0], rest_username, rest_password, self.configfile)
        status = subprocess.call(cmdstr, shell=True)
        if status != 0:
            raise Exception('Scan workload could not be applied')
        else:
            logger.info('Scan workload applied')

    def run(self):
        rmfile = "rm -f {}".format(self.test_config.stats_settings.secondary_statsfile)
        status = subprocess.call(rmfile, shell=True)
        if status != 0:
            raise Exception('existing 2i latency stats file could not be removed')
        else:
            logger.info('Existing 2i latency stats file removed')

        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        from_ts, to_ts = self.build_secondaryindex()
        self.access_bg()
        self.apply_scanworkload()
        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                *self.metric_helper.calc_secondaryscan_latency(percentile=80)
            )


class SecondaryIndexingScanLatencyRebalanceTest(SecondaryIndexingScanLatencyTest):

    """
    The test applies scan workload against the 2i server and measures
    and reports the average scan throughput
    """

    def rebalance(self, initial_nodes, nodes_after):
        clusters = self.cluster_spec.yield_clusters()
        for _, servers in clusters:
            master = servers[0]
            new_nodes = []
            ejected_nodes = []
            new_nodes = enumerate(
                servers[initial_nodes:nodes_after],
                start=initial_nodes
            )
            known_nodes = servers[:nodes_after]
            for i, host_port in new_nodes:
                self.rest.add_node(master, host_port)
        self.rest.rebalance(master, known_nodes, ejected_nodes)

    def run(self):
        rmfile = "rm -f {}".format(self.test_config.stats_settings.secondary_statsfile)
        status = subprocess.call(rmfile, shell=True)
        if status != 0:
            raise Exception('existing 2i latency stats file could not be removed')
        else:
            logger.info('Existing 2i latency stats file removed')

        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        initial_nodes = []
        nodes_after = [0]
        initial_nodes = self.test_config.cluster.initial_nodes
        nodes_after[0] = initial_nodes[0] + 1
        from_ts, to_ts = self.build_secondaryindex()
        self.access_bg()
        self.rebalance(initial_nodes[0], nodes_after[0])
        self.apply_scanworkload()
        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                *self.metric_helper.calc_secondaryscan_latency(percentile=80)
            )


class SecondaryIndexingLatencyTest(SecondaryIndexTest):

    """
    This test applies scan workload against a 2i server and measures
    the indexing latency
    """

    @with_stats
    def apply_scanworkload(self):
        rest_username, rest_password = self.cluster_spec.rest_credentials
        logger.info('Initiating the scan workload')
        cmdstr = "cbindexperf -cluster {} -auth=\"{}:{}\" -configfile scripts/config_indexinglatency.json -resultfile result.json".format(self.index_nodes[0], rest_username, rest_password)
        status = subprocess.call(cmdstr, shell=True)
        if status != 0:
            raise Exception('Scan workload could not be applied')
        else:
            logger.info('Scan workload applied')
        return status

    def run(self):
        self.load()
        self.wait_for_persistence()
        self.compact_bucket()
        self.hot_load()
        self.build_secondaryindex()
        num_samples = 100
        samples = []

        while num_samples != 0:
            access_settings = self.test_config.access_settings
            self.worker_manager.run_workload(access_settings, self.target_iterator)
            self.worker_manager.wait_for_workers()
            time_before = time.time()
            status = self.apply_scanworkload()
            time_after = time.time()
            if status == 0:
                num_samples = num_samples - 1
                time_elapsed = (time_after - time_before) / 1000000.0
                samples.append(time_elapsed)

        temp = np.array(samples)
        indexing_latency_percentile_80 = np.percentile(temp, 80)

        logger.info('Indexing latency (80th percentile): {} ms.'.format(indexing_latency_percentile_80))

        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(indexing_latency_percentile_80)
