# Changelog

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
