# Changelog

## milvus-sdk-cpp 2.5.4 (2026-01-20)
### Feature
- Add CreateSimpleCollectionRequest for quickly creating a simple collection
- Add Get() interface
- Compatible with zilliz cloud instance, support connecting with uri

### Improvement
- Pass collection_name/description to CreateCollectionRequest instead of CollectionSchema
- Pass num_shards to CreateCollectionRequest instead of CollectionSchema
- DescribeIndex/DropIndex support both field_name and index_name
- Add multi-vector methods for SearchRequest/SubSearchRequest
- Refine the request classes to reduce duplicate source code
- SearchResult can return recalls if enable_recall_calculation is true for zilliz cloud instance

### Bug
- AlterIndexProperties/DropIndexProperties cannot handle field_name, use index_name as input
- Fix a bug that CreateIndexRequest sync mode is not correctly work to wait index
- Fix a bug that SearchRequest.AddSparseVector() miss sparse vector when the size is unequal to the first vector
- Fix a bug that search response might treat collection name as primary key name

## milvus-sdk-cpp 2.5.3 (2025-12-22)
### Feature
- Support CreateCollection with indexes

### Improvement
- Replace AddXXXVector(fieldName, vector) of SearchRequest/SubSearchRequest with WithFieldName() + AddXXXVector(vector)
- LoadCollection/LoadPartitions with timeout setting, for waiting loading progress
- CreateIndex with timeout setting, for waiting index progress
- Flush with timeout setting, for waiting flush action complete
- DescribeCollection outputs properties
- GetLoadState returns loading progress percent
- Change default value of enable_dynamic_schema from false to true

### Bug
- Fix a bug that DescribeIndex() always returns all indexes no matter field_name is specified or not

## milvus-sdk-cpp 2.5.2 (2025-12-04)
### Feature
- Introduce MilvusClientV2 for clean/expanable interfaces

## milvus-sdk-cpp 2.5.1 (2025-11-21)
### Feature
- Support RunAnalyzer()

## milvus-sdk-cpp 2.5.0 (2025-11-12)
### Feature
- Support nullable field
- Support default value for field
- Support clustering key
- Support BM25 full text search
- Support text match
- Support BITMAP/HNSW_SQ/HNSW_PQ/HNSW_PRQ index types

### Improvement
- Support group_size and strict_group_size for Search() and HybridSearch()


## milvus-sdk-cpp 2.4.1 (2025-11-11)
### Feature
- Support partition key

### Bug
- Fix a bug that dynamic fields are missed for row-based insert/upsert
- Fix a bug that all dynamic fields are displayed for search/query even if the output_fields only contain some of them
- Fix a bug of dynamic field that describeCollection missed enable_dynamic
- Fix a bug for insert that incorrect range check of float values

### Improvement
- Allow to insert dynamic fields for column-based insert/upsert
- Change default value of shards_num to 1


## milvus-sdk-cpp 2.4.0 (2025-09-19)
### Feature
- Add new index types(DISKANN/AUTOINDEX/SCANN/GPU_IVF_FLAT/GPU_IVF_PQ/GPU_BRUTE_FORCE/GPU_CAGRA/TRIE/STL_SORT/INVERTED/SPARSE_INVERTED_INDEX/SPARSE_WAND)
- Add new metric types(COSINE)
- Support CreateDatabase/DropDatabase/ListDatabases/AlterDatabase/DescribeDatabase/UsingDatabase interfaces
- Support JSON field
- Support Array field
- Support dynamic field
- Support SparseVector field
- Support Float16Vector/BFloat16Vector field
- Support Upsert interface
- Support Consistency level
- Support query by count(*)
- Support GetLoadState interface
- Support HybridSearch interface
- Support row-based insert/upsert
- Support DescribeAlias/ListAliases interfaces
- Support AlterCollection/AlterCollectionField interfaces
- Support AlterIndexProperties/DropIndexProperties interfaces
- Support CreateResourceGroup/DropResourceGroup/UpdateResourceGroups/ListResourceGroups/DescribeResourceGroup interfaces
- Support TransferNode/TransferReplica interfaces
- Support CreateUser/UpdatePassword/DropUser/DescribeUser/ListUsers interfaces
- Support CreateRole/DropRole/DescribeRole/ListRoles/GrantRole/RevokeRole interfaces
- Support GrantPrivilege/RevokePrivilege/CreatePrivilegeGroup/DropPrivilegeGroup/ListPrivilegeGroups/AddPrivilegesToGroup/RemovePrivilegesFromGroup interfaces
- Support QueryIterator/SearchIterator
- Support CreateImportJobs/ListImportJobs/GetImportJobProgress interfaces


### Improvement
- Support connection with dbname
- DropIndex() interface accepts index_name instead of field_name
- Cache collection schema to reduce DescribeCollection call
- Support search with grouping by field
- Add retry machinery for rpc call
- Add CurrentUsedDatabase method for MilvusClient
- Add methods for QuertResults/SearchResults to return row-based results


### Orther changes
- Remove unsupported index types(IVF_HNSW/RHNSW_FLAT/RHNSW_SQ/RHNSW_PQ/ANNOY)
- Remove unsupported metric types(TANIMOTO/SUBSTRUCTURE/SUPERSTRUCTURE)
- Remove CalcDistance interface, not supported by milvus 2.4
- Remove Flush call in GetCollectionStats/GetPartitionStats/CreateIndex, milvus 2.4 doesn't allow calling Flush constantly
- Directly pass radius/range_filter for range search, no need to verify them on the client side
- CreateIndex() internally calls DescribeIndex() to check index state instead of GetIndexState()
- Remove client-side checks for index parameters and search parameters, let milvus server validate
- Deprecate ShowCollections, replaced by ListCollections
- Deprecate ShowPartitions, replaced by ListPartitions
- Rename GetVersion to GetServerVersion
- Link gRPC as shared library instead of static link
- Optional to build with external pre-installed gRPC lib
- Add more examples
- Simplify the project to reduce dependencies, only one dependency(gRPC) now
