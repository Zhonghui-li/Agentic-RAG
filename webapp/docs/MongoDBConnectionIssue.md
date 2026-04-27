# MongoDB Connection Issue Troubleshooting

## Issue

Encountering `MongoNetworkError: connect ECONNREFUSED 127.0.0.1:27017` when trying to connect to MongoDB server.

## Steps to Resolve

### Step 1: Check for Lock File

First, check if there is a lock file that might be blocking the MongoDB server:

```sh
ls /tmp/mongodb-27017*
```

Output:

```sh
/tmp/mongodb-27017.sock
```

### Step 2: Remove the Lock File

Remove the lock file if it exists. [Here is the reference](https://www.mongodb.com/community/forums/t/connect-econnrefused-127-0-0-1-27017-in-mongodb-compass/166773):

```sh
rm /tmp/mongodb-27017*
```

### Step 3: Verify Removal

Check again to ensure the lock file is removed:

```sh
ls /tmp/mongodb-27017*
```

Output:

```sh
zsh: no matches found: /tmp/mongodb-27017*
```

### Step 4: Restart MongoDB Server

Restart the MongoDB server. Since the `systemctl` and `service` commands were not found, use Homebrew to restart the MongoDB service:

```sh
brew services restart mongodb-community
```

Output:

```sh
Stopping `mongodb-community`... (might take a while)
==> Successfully stopped `mongodb-community` (label: homebrew.mxcl.mongodb-community)
==> Successfully started `mongodb-community` (label: homebrew.mxcl.mongodb-community)
```

### Step 5: Verify MongoDB Service Status

Check the status of the MongoDB service to ensure it's running:

```sh
brew services list | grep mongodb
```

Output:

```sh
mongodb-community     started         yourusername ~/Library/LaunchAgents/homebrew.mxcl.mongodb-community.plist
mongodb-community@5.0 error  12288    yourusername ~/Library/LaunchAgents/homebrew.mxcl.mongodb-community@5.0.plist
mongodb-community@6.0 error  15872    yourusername ~/Library/LaunchAgents/homebrew.mxcl.mongodb-community@6.0.plist
```

### Step 6: Connect Using `mongosh`

Attempt to connect to MongoDB using `mongosh`:

```sh
mongosh
```

Output:

```sh
Current Mongosh Log ID:	669ea439959e61bf5625284e
Connecting to:		mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.12
Using MongoDB:		7.0.12
Using Mongosh:		2.2.12

For mongosh info see: https://docs.mongodb.com/mongodb-shell/

------
   The server generated these startup warnings when booting
   2024-07-22T11:25:42.413-07:00: Access control is not enabled for the database. Read and write access to data and configuration is unrestricted
------
```

### Step 7: Exit `mongosh`

Exit the `mongosh` shell:

```sh
test> exit
```

## Notes

- Ensure MongoDB is properly installed and configured.
- Consider enabling access control for the database to restrict read and write access to data and configuration.

By following these steps, the connection issue with MongoDB should be resolved.
