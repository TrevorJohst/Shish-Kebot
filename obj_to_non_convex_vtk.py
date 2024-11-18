import pyvista
import tetgen

filename = 'skewer_5mm.obj'
reader = pyvista.get_reader(filename)
mesh = reader.read()
mesh.save('non_convex_mesh.ply', binary=False)

tgen = tetgen.TetGen('non_convex_mesh.ply')
# Same as this command-line option: tgen.tetrahedralize(switches='qYMkV')
tgen.tetrahedralize(quality=True, nobisect=True, nomergefacet=True,
                    nomergevertex=True, vtksurfview=True, vtkview=True,
                    verbose=True)

tgen.write('non_convex_mesh.vtk', binary=False)
