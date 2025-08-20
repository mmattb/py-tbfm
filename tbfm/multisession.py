# Okay; this is a bit complicated... We want to be able to build the computational graph depending on provided options:
#  Session specific normalizers
#  Session-shared but TTA'd AEs versus session-specific AEs
#  Session-shared and TTA'd TBFMs versus session-specific TBFMs.
# We need a protocol whereby we input batches which have potentially mixed sessions, runs through the graph, and gives outputs, regardless
#  of the topology options above...
# In general: x and y values will be packaged as a list: (session_id, *args, **kwargs)


def build(session_data):
"""
session_data: [(batch, time, ch), (batch_time-ch
"""
