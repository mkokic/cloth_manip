from data_utils import load_data
from IPython import embed

path = '/home/melhua/code/pointnet/data/modelnet40_ply_hdf5_2048_pts'
data_train, data_test = load_data(2048)
# for i, mesh in enumerate(data_train['pInit']):
#     pts = mesh.reshape(2048, 3)
#     f = open(path + '/train/mesh_%d.xyz' % i, 'w')
#     for p in pts:
#         pts_text = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n'
#         f.writelines(pts_text)
#     f.close()
for i, mesh in enumerate(data_test['pInit']):
    pts = mesh.reshape(2048, 3)
    f = open(path + '/test/mesh_%d.xyz' % i, 'w')
    for p in pts:
        pts_text = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n'
        f.writelines(pts_text)
    f.close()

