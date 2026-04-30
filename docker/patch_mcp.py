import site, os
OLD = 'from mcp.client.streamable_http import streamable_http_client'
NEW = 'from mcp.client.streamable_http import streamablehttp_client as streamable_http_client'
for sp in site.getsitepackages():
    f = os.path.join(sp, 'openharness', 'mcp', 'client.py')
    if os.path.exists(f):
        txt = open(f).read()
        if OLD in txt:
            open(f, 'w').write(txt.replace(OLD, NEW))
            print('Patched', f)
        else:
            print('Already patched or not found in', f)
