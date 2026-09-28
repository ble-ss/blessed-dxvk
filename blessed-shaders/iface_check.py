# blessed: shader-replace, static interface check of a replacement dxbc against the vanilla one
"""
usage: python iface_check.py VANILLA.dxbc REPLACEMENT.dxbc

disassembles both with fxc /dumpbin and compares what the pipeline sees:
  - shader type
  - input and output declarations (dcl_input*, dcl_output*): must match
  - resources, samplers, uavs: every one the replacement declares must be
    declared by vanilla with the same slot and the same declaration
  - constant buffers: the replacement's slots must be vanilla's, and no
    larger than vanilla's
  - thread group size (compute): reported; a change is legal only if the
    replacement covers the same output tile per group (the dispatch count
    is the game's)
exit status 0 = pass, 1 = refused.
"""
import os
import re
import subprocess
import sys
import tempfile

FXC = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.28000.0\x64\fxc.exe"


def disasm(path):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "o.asm")
        subprocess.run([FXC, "/nologo", "/dumpbin", path, "/Fc", out], check=True, capture_output=True)
        with open(out) as f:
            return [l.rstrip() for l in f if l.strip() and not l.startswith("//")]


def decls(lines):
    kind = lines[0]
    io, res, cbs, group = set(), {}, {}, None
    for l in lines[1:]:
        if not l.startswith("dcl_"):
            continue
        if l.startswith(("dcl_input", "dcl_output")):
            # compute thread ids are not an interface; a rewrite may read others
            if "vThread" not in l:
                io.add(re.sub(r"\s+", " ", l))
        elif l.startswith("dcl_constantbuffer"):
            m = re.match(r"dcl_constantbuffer CB(\d+)\[(\d+)\]", l)
            cbs[int(m.group(1))] = int(m.group(2))
        elif l.startswith(("dcl_resource", "dcl_sampler", "dcl_uav")):
            m = re.search(r"\b([tsu]\d+)\b", l)
            res[m.group(1)] = re.sub(r"\s+", " ", l)
        elif l.startswith("dcl_thread_group"):
            group = l.split(" ", 1)[1]
    return kind, io, res, cbs, group


def main(vpath, rpath):
    vk, vio, vres, vcb, vg = decls(disasm(vpath))
    rk, rio, rres, rcb, rg = decls(disasm(rpath))
    errors, notes = [], []
    if vk != rk:
        errors.append("shader type %s vs %s" % (vk, rk))
    if vio != rio:
        errors.append("i/o differs: only vanilla %s, only replacement %s" % (sorted(vio - rio), sorted(rio - vio)))
    for slot, d in rres.items():
        if vres.get(slot) != d:
            errors.append("%s: %s (vanilla: %s)" % (slot, d, vres.get(slot)))
    for slot, size in rcb.items():
        if slot not in vcb:
            errors.append("cb%d not declared by vanilla" % slot)
        elif size > vcb[slot]:
            errors.append("cb%d: %d vectors > vanilla %d" % (slot, size, vcb[slot]))
    unused = sorted(set(vres) - set(rres)) + ["cb%d" % s for s in sorted(set(vcb) - set(rcb))]
    if unused:
        notes.append("vanilla bindings the replacement no longer reads: %s" % ", ".join(unused))
    if vg != rg:
        notes.append("thread group %s -> %s" % (vg, rg))
    name = os.path.basename(rpath)
    for n in notes:
        print("%s: note: %s" % (name, n))
    for e in errors:
        print("%s: REFUSED: %s" % (name, e))
    if not errors:
        print("%s: interface ok" % name)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
