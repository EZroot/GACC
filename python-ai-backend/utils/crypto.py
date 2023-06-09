import hashlib

def calculate_hash(string):
    # Create a SHA256 hash object
    hash_object = hashlib.sha256()

    # Convert the string to bytes and update the hash object
    hash_object.update(string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hash_value = hash_object.hexdigest()

    return hash_value
