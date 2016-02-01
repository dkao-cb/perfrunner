[clusters]
ycsb_small =
    172.23.106.160:8091
    172.23.105.248:8091
    172.23.106.48:8091
    172.23.106.47:8091

[clients]
hosts =
    172.23.99.26
credentials = root:couchbase

[storage]
data = /data
index = /data

[credentials]
rest = Administrator:password
ssh = root:couchbase

[parameters]
Platform = VM
