import msgpack
import binascii

# This script deserializes several msgpack blobs and prints their contents and types.
# Used for debugging deserialization issues.

# The hex blobs from checkpoint_writes - as actual hex strings (no \x prefix)
blobs = {
    "messages": "",
    "prompt": "",
    "context": "",
    "tags": "",
    "mcp_responses": ""
}

for name, hex_str in blobs.items():
    try:
        # Convert hex string to binary bytes
        data = binascii.unhexlify(hex_str)
        print(f"\n{name}:")
        print(f"  Bytes: {len(data)} total")
        print(f"  First 20 bytes (hex): {data[:20].hex()}")
        
        # Unpack the msgpack data
        result = msgpack.unpackb(data, raw=False)
        print(f"  Type: {type(result).__name__}")
        print(f"  Value: {result}")
                
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")
