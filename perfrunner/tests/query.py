from perfrunner.helpers.cbmonitor import with_stats
from perfrunner.tests.index import IndexTest, DevIndexTest


class QueryTest(IndexTest):

    COLLECTORS = {'latency': True, 'query_latency': True}

    @with_stats
    def access(self):
        super(QueryTest, self).timer()

    def run(self):
        self.load()
        self.wait_for_persistence()

        self.compact_bucket()

        self.hot_load()

        self.define_ddocs()
        self.build_index()

        self.workload = self.test_config.access_settings
        self.access_bg()
        self.access()


class QueryThroughputTest(QueryTest):

    def run(self):
        super(QueryThroughputTest, self).run()
        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                self.metric_helper.calc_avg_couch_views_ops()
            )


class QueryLatencyTest(QueryTest):

    def run(self):
        super(QueryLatencyTest, self).run()

        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                *self.metric_helper.calc_query_latency(percentile=80)
            )
            if self.remote.os != 'Cygwin' and \
                    self.test_config.stats_settings.post_rss:
                self.reporter.post_to_sf(
                    *self.metric_helper.calc_max_beam_rss()
                )


class IndexLatencyTest(QueryTest):

    COLLECTORS = {'index_latency': True, 'query_latency': True}

    def run(self):
        super(IndexLatencyTest, self).run()

        if self.test_config.stats_settings.enabled:
            self.reporter.post_to_sf(
                *self.metric_helper.calc_observe_latency(percentile=95)
            )


class DevQueryLatencyTest(DevIndexTest, QueryLatencyTest):

    pass
