import torch
from torch.utils import data
from torch.nn import functional as F
from tqdm import tqdm

from config import *
from data_generator import DataGenerator, Dataset
from network import PretrainNet
from utils import *

def test(model, data_loader, threshold=0.5):
	model.eval()
	result = []
	# result_s = []
	with torch.no_grad():
		for data in tqdm(data_loader):
			node_onehot, node_pe, pred_pairs, pred_pair_num, pad_mask, contact_map, seq_length, _ = data
			node_onehot = node_onehot.cuda()
			node_pe = node_pe.cuda()
			pred_pairs = pred_pairs.cuda()
			pred_pair_num = pred_pair_num.cuda()
			pad_mask = pad_mask.cuda()
			contact_map = contact_map.cuda()
			seq_length = seq_length.cuda()

			input = (node_onehot, node_pe, pred_pairs, pad_mask)

			pred = model(input)

			# retain high probability
			edge_prob = F.sigmoid(pred)
			base_pairs = edge_prob

			cum = 0
			offset = 0
			offset_cm = 0
			for i in range(len(seq_length)):
				contact_map_pred = torch.zeros((seq_length[i], seq_length[i])).cuda()
				idx = pred_pairs[:, cum:cum+pred_pair_num[i]] - offset
				contact_map_pred[idx[0,:], idx[1,:]] = base_pairs[cum:cum+pred_pair_num[i]]
				contact_map_pred = contact_map_pred + contact_map_pred.T

				mask_matrix = torch.zeros([seq_length[i], seq_length[i]]).cuda()
				mask_matrix[idx[0], idx[1]] = 1
				mask_matrix = mask_matrix + mask_matrix.T

				contact_map_pred = contact_map_pred * mask_matrix.float()
				contact_map_pred = post_process_argmax(contact_map_pred, threshold)

				# param = (contact_map_pred, node_set1_list[i], threshold)
				# contact_map_pred = post_process_HK(param)

				contact_map_label = contact_map[offset_cm:offset_cm+seq_length[i]**2].cuda().reshape(seq_length[i], seq_length[i])

				result.append(evaluate(contact_map_pred, contact_map_label))
				# result_s.append(evaluate_shifted(contact_map_pred, contact_map_label))

				cum += pred_pair_num[i]
				offset += seq_length[i]
				offset_cm += seq_length[i]**2

	nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result)
	print('precision: ', np.average(nt_exact_p))
	print('recall: ', np.average(nt_exact_r))
	print('F1: ', np.average(nt_exact_f1))
	print()

	# nt_exact_p_s,nt_exact_r_s,nt_exact_f1_s = zip(*result_s)
	# print('precision_s: ', np.average(nt_exact_p_s))
	# print('recall_s: ', np.average(nt_exact_r_s))
	# print('F1_s: ', np.average(nt_exact_f1_s))
	# print()

	return np.average(nt_exact_f1), np.average(nt_exact_p), np.average(nt_exact_r)

def main():
	args = get_args()
	config_file = args.config
	config = process_config(config_file)
	print('Here is the configuration of this run: ')
	print(config)
	
	torch.cuda.set_device(0)
	threshold = config.threshold
	embedding_dim = config.embedding_dim
	ct_layer_num = config.layer_num
	nhead = config.nhead
	batch_size = 8
	num_workers = 8
	
	params = {'drop_last': False,
			  'collate_fn': collate_test}
	
	RNAStralign_test_data = DataGenerator(f'data/RNAStralign', f'test_max600', node_input_dim=embedding_dim, mode='test')
	RNAStralign_test_dataset = Dataset(RNAStralign_test_data)
	RNAStralign_test_loader = data.DataLoader(RNAStralign_test_dataset, 
									   **params, 
									   batch_size=16, 
									   num_workers=4, 
									   pin_memory=True,
									   shuffle=False)
	
	ArchiveII_data = DataGenerator(f'data/ArchiveII', f'all', node_input_dim=embedding_dim, mode='test')
	ArchiveII_dataset = Dataset(ArchiveII_data)
	ArchiveII_loader = data.DataLoader(ArchiveII_dataset, 
									   **params, 
									   batch_size=8, 
									   num_workers=8, 
									   pin_memory=True,
									   shuffle=False)
	
	bpRNA1m_test_data = DataGenerator(f'data/bpRNA_1m', f'test', node_input_dim=embedding_dim, mode='test')
	bpRNA1m_test_dataset = Dataset(bpRNA1m_test_data)
	bpRNA1m_test_loader = data.DataLoader(bpRNA1m_test_dataset, 
									   **params, 
									   batch_size=32, 
									   num_workers=8, 
									   pin_memory=True,
									   shuffle=False)
	
	bpRNAnew_data = DataGenerator(f'data/bpRNA_new', f'all', node_input_dim=embedding_dim, mode='test')
	bpRNAnew_dataset = Dataset(bpRNAnew_data)
	bpRNAnew_loader = data.DataLoader(bpRNAnew_dataset, 
									   **params, 
									   batch_size=batch_size, 
									   num_workers=num_workers, 
									   pin_memory=True,
									   shuffle=False)
	
	ArchiveII_families = ['5s', '23s', 'srp', 'telomerase', '16s', 'grp1', 'RNaseP', 'tmRNA', 'tRNA']
	family_loaders = [data.DataLoader(
						Dataset(DataGenerator(f'data/ArchiveII/family_fold', f'{family}', node_input_dim=embedding_dim, mode='family_fold')),
						**params, 
						batch_size=batch_size, 
						num_workers=num_workers, 
						pin_memory=True,
						shuffle=False)
						for family in ArchiveII_families]
	
	print('Data Loading Done!!!')
	
	model = PretrainNet(embedding_dim, ct_layer_num, nhead)
	checkpoint = torch.load('checkpoints/CaTFold_pretrain.pt')
	model.load_state_dict(checkpoint)
	model.cuda()
	print("Model #Params Num : %d" % (sum([x.nelement() for x in model.parameters()]),))

	print('RNAStralign test...') # 0.738
	test(model, RNAStralign_test_loader, threshold)

	print('ArchiveII...') # 0.690
	f1, _, _ = test(model, ArchiveII_loader, threshold)

	print('bpRNA-1m test...') # 0.547
	test(model, bpRNA1m_test_loader, threshold)

	print('bpRNAnew...') # 0.609
	f1, _, _ = test(model, bpRNAnew_loader, threshold)

	print('ArchiveII families...')
	f1_list = []
	for i, family in enumerate(ArchiveII_families):
		print(f'{family}...')
		f1, _, _ = test(model, family_loaders[i], threshold)
		f1_list.append(f1)
	print('ArchiveII family mean: ', np.mean(f1_list))
		
if __name__ == '__main__':
	"""
	See module-level docstring for a description of the script.
	"""
	main()