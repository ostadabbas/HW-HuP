'''
collect the result from the metrics
'''
import os
import os.path as osp
import json

# collection setting
def collecRGB():
	'''
	this collect RGB result
	:return:
	'''
	fNm = 'RGB_uncover'
	li_exp = [
		'SLP_2D_preT',
		'SLP_2D_e30',
		'SLP_2D_HMR_e30',
		'SLP_3D_e30',
		'SLP_3D_vis_e30',
		'SLP_3D_vis_d_e30',
		'SLP_3D_vis_d_op0_e30',
		'SLP_3D_vis_d_noOpt_e30',
	]
	mNm = 'err_depth'

	fd = 'metrics'
	if not osp.exists(fd):
		os.makedirs(fd)
	pth = osp.join(fd, fNm+'.txt')
	pth_ltx = osp.join(fd, fNm + '_ltx.txt')
	li_rst = []
	f_out= open(pth, 'w')
	f_ltx_out= open(pth_ltx, 'w')
	n_exp = len(li_exp)
	str_ltx = ''
	for exp in li_exp:
		pth_exp = osp.join('logs', exp, 'eval_metric.json')
		with open(pth_exp, 'r') as f:
			dt_in = json.load(f)
			metric_t = dt_in[mNm]
			li_rst.append(metric_t)
			f_out.write(str(metric_t))
			f_out.write('\t')
			# f_ltx_out.write(str(metric_t))
			# f_ltx_out.write('\&')
			# f_ltx_out.write('{:.2f} \&'.format(metric_t))
			str_ltx+= '{:.2f} & '.format(metric_t)
			f.close()

	# replace the last  \& to \\
	li_str_ltx = list(str_ltx)
	str_ltx = "".join(li_str_ltx[:-2])+r'\\'
	f_ltx_out.write(str_ltx)

	f_out.close()
	f_ltx_out.close()

	print(mNm)
	print(li_rst)

def collectAllC():
	fNm = 'RGB_uncover'
	li_exp = [
		'SLP_2D_{}_preT',
		'SLP_2D_{}_e30',
		'SLP_2D_HMR_{}_e30',
		# 'SLP_3D_e30',
		# 'SLP_3D_vis_e30',
		'SLP_3D_vis_d_{}_e30',
		# 'SLP_3D_vis_d_op0_e30',
		'SLP_3D_vis_d_noOpt_{}_e30',
	]
	mNm = 'err_depth'
	mod_li = ['depth', 'IR', 'PM', 'stk']
	fd = 'metrics'
	cov_nms = ['uncover', 'cover1', 'cover2', 'allC']   # the cover names,   outer loop
	if not osp.exists(fd):
		os.makedirs(fd)
	for i_c, cov in enumerate(cov_nms):
		sv_pth = osp.join(fd, cov+'_ltx.txt')       # save only latex version at this time , just add other with \t for excel .
		str_sv = ''  # write this string
		for exp in li_exp:
			for mod in mod_li:
				pth_f = osp.join('logs',exp.format(mod), 'eval_metric.json')
				with open(pth_f, 'r') as f:
					dt_in = json.load(f)
					ed_c = dt_in['err_depth_cat']
					ed_c.append(dt_in['err_depth'])
					f.close()
				rst_t  = ed_c[i_c]      # whith case
				str_sv+= '{:.2f} & '.format(rst_t)   # single entry of m_mod from the c
			str_sv = str_sv[:-2]+r'\\'+ '\n'  # turn ling

		with open(sv_pth, 'w') as f:
			f.write(str_sv)
			f.close()

if __name__ == '__main__':
	# collecRGB()       # for RGB version
	collectAllC()



