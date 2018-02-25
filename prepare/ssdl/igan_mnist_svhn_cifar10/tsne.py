import scipy.io as sio
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
args = parser.parse_args()
print(args)

# read data
data = sio.loadmat(args.data_path + '/fea.mat')
trainx = data['trainx']
trainy = data['trainy']
testx = data['testx']
testy = data['testy']

# TSNE
trainx_embedded = TSNE(n_components=2).fit_transform(trainx)
testx_embedded = TSNE(n_components=2).fit_transform(testx)

sio.savemat(args.data_path + '/fea_tsne.mat', 
            mdict={'trainx': trainx_embedded, 'trainy': trainy,
                   'testx': testx_embedded, 'testy': testy})