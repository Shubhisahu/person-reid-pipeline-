import json

with open('D:/New folder (3)/reid_pipeline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
code_cells  = [c for c in cells if c['cell_type'] == 'code']
md_cells    = [c for c in cells if c['cell_type'] == 'markdown']
total_lines = sum(len(c['source']) for c in code_cells)

print(f'nbformat          : {nb["nbformat"]}')
print(f'Total cells       : {len(cells)}')
print(f'Code cells        : {len(code_cells)}')
print(f'Markdown cells    : {len(md_cells)}')
print(f'Total code lines  : {total_lines}')
print(f'File size         : {len(json.dumps(nb, ensure_ascii=False))/1024:.1f} KB')
print()

for i, c in enumerate(cells):
    assert c.get('cell_type') in ('code', 'markdown'), f'Cell {i} bad type'
    assert isinstance(c.get('source'), list), f'Cell {i} source not list'

print('All cells valid JSON structure.')
print()
for i, c in enumerate(cells):
    src = ''.join(c['source'])[:70].strip().replace('\n', ' ')
    print(f'  [{i:02d}] {c["cell_type"]:8s} {src}')

nb_text = json.dumps(nb)
checks = [
    ('FIX 1 - market1501 path',    'market1501'),
    ('FIX 2 - GIF highlight',      'cyan border'),
    ('FIX 3 - leakage assert',     'Identity leakage detected'),
    ('FIX 4 - multi_cam_pids',     'multi_cam_pids'),
    ('FIX 5 - threshold param',    'threshold'),
    ('FIX 6 - 500-query subset',   'N_RR'),
]

print()
for label, keyword in checks:
    found = keyword in nb_text
    status = 'PASS' if found else 'FAIL'
    print(f'  [{status}] {label}')
