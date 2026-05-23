content = open(r'F:\BaiduNetdiskDownload\project\local-RAG\static\js\app.js', 'r', encoding='utf-8').read()
depth = 0
min_depth = 0
min_line = 0
for i, c in enumerate(content):
    if c == '{':
        depth += 1
    elif c == '}':
        depth -= 1
        if depth < min_depth:
            min_depth = depth
            min_line = content[:i].count('\n') + 1
print(f'Minimum depth reached: {min_depth} at line {min_line}')
# Show around that line
lines = content.split('\n')
for ln in range(max(0, min_line-3), min(len(lines), min_line+2)):
    print(f'{ln+1:4d}: {lines[ln]}')
