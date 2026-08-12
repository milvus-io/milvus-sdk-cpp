# Tutorial 8: Manage users and roles (advanced)

This tutorial demonstrates Milvus role-based access control (RBAC) with `MilvusClientV2`. It creates
a temporary user, role, and custom privilege group, grants query/search privileges on all collections
in the `default` database, describes the user, and removes every resource it created.

## API flow

1. `CreateUser` creates a login identity.
2. `CreateRole` creates a permission container.
3. `CreatePrivilegeGroup` and `AddPrivilegesToGroup` define reusable permissions.
4. `GrantPrivilegeV2` scopes the group to a database and collection pattern.
5. `GrantRole` assigns the role to the user; `DescribeUser` reads the assignment.
6. Drop the user, role, and privilege group during cleanup.

## Prerequisites and run

Milvus 2.6 or later must have authorization enabled, and `MILVUS_TOKEN` must belong to an
administrator allowed to manage users, roles, and privileges. Defaults are
`MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

Set `MILVUS_USER_PASSWORD` to choose the password for the temporary user. It defaults to
`CppTutorial!123` for local tutorial use.

```bash
make
MILVUS_USER_PASSWORD="a-strong-temporary-password" make run
```

The tutorial uses fixed temporary names (`cpp_tutorial_*`), so do not run concurrent copies with the
same server credentials. It removes those resources on normal completion. The default Conan package
is `milvus-sdk-cpp/3.0.2@milvus/dev`; override it with `MILVUS_SDK_*` variables.

## Expected output

```text
Calling Connect...
Connect succeeded.
Calling DropUser for stale data...
DropUser stale-data cleanup completed.
Calling DropRole for stale data...
DropRole stale-data cleanup completed.
Calling DropPrivilegeGroup for stale data...
DropPrivilegeGroup stale-data cleanup completed.
Calling CreateUser...
CreateUser succeeded.
Calling CreateRole...
CreateRole succeeded.
Calling CreatePrivilegeGroup...
CreatePrivilegeGroup succeeded.
Calling AddPrivilegesToGroup...
AddPrivilegesToGroup succeeded.
Calling GrantPrivilegeV2...
GrantPrivilegeV2 succeeded.
Calling GrantRole...
GrantRole succeeded.
Calling DescribeUser...
DescribeUser succeeded.
User: cpp_tutorial_user, roles=1
Calling DropUser...
DropUser succeeded.
Calling DropRole...
DropRole succeeded.
Calling DropPrivilegeGroup...
DropPrivilegeGroup succeeded.
```

## Troubleshooting

- RBAC management errors usually mean `MILVUS_TOKEN` lacks administrator privileges.
- authorization must be enabled for grants to be enforced.
- `connection refused`: start Milvus or set `MILVUS_URI` to a reachable endpoint.
- rerunning after interruption: best-effort cleanup removes the fixed user, role, and group names first.
