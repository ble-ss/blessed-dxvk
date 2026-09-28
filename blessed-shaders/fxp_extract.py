# blessed: shader-replace, pulls vanilla dxbc blobs out of skyrim's shader bundle by hash
"""
usage: python fxp_extract.py OUTDIR HASH [HASH ...]
       python fxp_extract.py --map MAP.json

reads Data/Skyrim - Shaders.bsa (read only; an uncompressed v105 bsa holding
one file, shadersfx/shaders011.fxp) and scans the fxp for DXBC containers.
a blob's name is dxvk's: "<stage>.<32 hex>", the hex being the container's
own md5 checksum (bytes 4-19). HASH may be a full name, 32 hex digits or an
8+ digit prefix. --map writes every name -> {offset, size} as json.

the fxp's own record layout (technique ids per blob) is not parsed yet; the
scan relies on the DXBC magic and the container size field only.
"""
import json
import os
import struct
import sys

BSA = r"D:\SteamLibrary\steamapps\common\Skyrim Special Edition\Data\Skyrim - Shaders.bsa"
STAGES = {0: "fs", 1: "vs", 2: "gs", 3: "tcs", 4: "tes", 5: "cs"}


def read_fxp(path=BSA):
    with open(path, "rb") as f:
        magic, ver, off, flags, nfold, nfile, lfold, lfile, ftype = struct.unpack("<4sIIIIIIII", f.read(36))
        if magic != b"BSA\0" or ver != 105 or nfile != 1 or flags & 0x4:
            raise SystemExit("unexpected bsa layout (want v105, one uncompressed file)")
        f.read(24)  # the one folder record
        n = f.read(1)[0]
        f.read(n)  # folder name
        _, size, doff = struct.unpack("<QII", f.read(16))
        f.seek(doff)
        return f.read(size & 0x3FFFFFFF)


def program_type(blob):
    count = struct.unpack_from("<I", blob, 28)[0]
    for i in range(count):
        o = struct.unpack_from("<I", blob, 32 + 4 * i)[0]
        if blob[o:o + 4] in (b"SHEX", b"SHDR"):
            return struct.unpack_from("<I", blob, o + 8)[0] >> 16
    return None


def scan(data):
    out = {}
    p = 0
    while True:
        p = data.find(b"DXBC", p)
        if p < 0:
            return out
        size = struct.unpack_from("<I", data, p + 24)[0]
        blob = data[p:p + size]
        if len(blob) == size and size > 32:
            stage = STAGES.get(program_type(blob), "shdr")
            name = "%s.%s" % (stage, blob[4:20].hex())
            out.setdefault(name, (p, size))
        p += 4


def main(argv):
    data = read_fxp()
    blobs = scan(data)
    if argv[:1] == ["--map"]:
        with open(argv[1], "w") as f:
            json.dump({k: {"offset": v[0], "size": v[1]} for k, v in sorted(blobs.items())}, f, indent=0)
        print("%d blobs -> %s" % (len(blobs), argv[1]))
        return
    outdir = argv[0]
    os.makedirs(outdir, exist_ok=True)
    for want in argv[1:]:
        hexpart = want.split(".")[-1]
        hits = [k for k in blobs if k.split(".")[1].startswith(hexpart)]
        if len(hits) != 1:
            print("%s: %d matches, skipped" % (want, len(hits)))
            continue
        off, size = blobs[hits[0]]
        with open(os.path.join(outdir, hits[0] + ".dxbc"), "wb") as f:
            f.write(data[off:off + size])
        print("%s -> %s.dxbc (%d bytes)" % (want, hits[0], size))


if __name__ == "__main__":
    main(sys.argv[1:])
