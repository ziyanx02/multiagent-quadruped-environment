# import ptvsd
import debugpy
import sys

def break_into_debugger(port= 6789):
    ip_address = ('0.0.0.0', port)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()
