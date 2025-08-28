import os
import sys
import subprocess

old = bytes.fromhex(sys.argv[2])
new = bytes.fromhex(sys.argv[3])

if len(new) != len(old):
    raise SystemExit("New hash must be exactly the same length.")

chunk_size = 1024 * 1024  # 1 MB
found_offset = None

os.chdir(sys.argv[1])

subprocess.run("cp system.physmem.store0.pmem checkpoint.gz", shell=True, check=True)
subprocess.run("gunzip checkpoint.gz", shell=True, check=True)

with open("./checkpoint", "r+b") as f:
    overlap = len(old) - 1  # to handle matches across chunk boundaries
    pos = 0
    prev_chunk = b""
    
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        
        data = prev_chunk + chunk
        idx = data.find(old)
        if idx != -1:
            found_offset = pos - len(prev_chunk) + idx
            f.seek(found_offset)
            f.write(new)
            print(f"Patched at offset 0x{found_offset:08x}")
            break
        
        pos += len(chunk)
        prev_chunk = chunk[-overlap:]

if found_offset is None:
    print("Hash not found.")
    exit(1)

subprocess.run("mv system.physmem.store0.pmem system.physmem.store0.pmem.old", shell=True, check=True)
subprocess.run("gzip checkpoint", shell=True, check=True)
subprocess.run("mv checkpoint.gz system.physmem.store0.pmem", shell=True, check=True)
