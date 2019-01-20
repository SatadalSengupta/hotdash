import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


RESULTS_FOLDER = './results/'


def main():
	total_reward_all = {}
	total_reward_all['BB'] = {}
	total_reward_all['RB'] = {}
	total_reward_all['FIXED'] = {}
	total_reward_all['FESTIVE'] = {}
	total_reward_all['BOLA'] = {}
	total_reward_all['RL'] = {}
	total_reward_all['robustMPC'] = {}
	total_reward_all['fastMPC'] = {}

	log_files = os.listdir(RESULTS_FOLDER)

	for log_file in log_files:

		R = 0
		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			lines = [line.rstrip() for line in f.readlines() if line.rstrip()]
			
			for line in lines:
				parse = line.split()
				R += float(parse[-1])
		
		# break

		if 'BB' in log_file:
			total_reward_all['BB'][log_file[7:]] = R
		elif 'RB' in log_file:
			total_reward_all['RB'][log_file[7:]] = R
		elif 'FIXED' in log_file:
			total_reward_all['FIXED'][log_file[10:]] = R
		elif 'FESTIVE' in log_file:
			total_reward_all['FESTIVE'][log_file[12:]] = R
		elif 'BOLA' in log_file:
			total_reward_all['BOLA'][log_file[9:]] = R
		elif 'RL' in log_file:
    			total_reward_all['RL'][log_file[7:]] = R
		elif 'robustMPC' in log_file:
    			total_reward_all['robustMPC'][log_file[14:]] = R
		elif 'fastMPC' in log_file:
    			total_reward_all['fastMPC'][log_file[12:]] = R

		else:
			print "Error: log name doesn't contain proper abr schemes."
		
		
	log_file_all = []
	BB_reward_all = []
	RB_reward_all = []
	FIXED_reward_all = []
	FESTIVE_reward_all = []
	BOLA_reward_all = []
	RL_reward_all = []
	robustMPC_reward_all = []
	fastMPC_reward_all = []

	for l in total_reward_all['BB']:
    		print l
		if l in total_reward_all['RB'] and \
		   l in total_reward_all['FIXED'] and \
		   l in total_reward_all['FESTIVE'] and \
		   l in total_reward_all['BOLA'] and \
		   l in total_reward_all['RL'] and \
		   l in total_reward_all['robustMPC'] and \
		   l in total_reward_all['fastMPC']:
				log_file_all.append(l)
				BB_reward_all.append(total_reward_all['BB'][l])
				RB_reward_all.append(total_reward_all['RB'][l])
				FIXED_reward_all.append(total_reward_all['FIXED'][l])
				FESTIVE_reward_all.append(total_reward_all['FESTIVE'][l])
				BOLA_reward_all.append(total_reward_all['BOLA'][l])
				RL_reward_all.append(total_reward_all['RL'][l])
				robustMPC_reward_all.append(total_reward_all['robustMPC'][l])
				fastMPC_reward_all.append(total_reward_all['fastMPC'][l])

	BB_total_reward = np.mean(BB_reward_all)
	RB_total_reward = np.mean(RB_reward_all)
	FIXED_total_reward = np.mean(FIXED_reward_all)
	FESTIVE_total_reward = np.mean(FESTIVE_reward_all)
	BOLA_total_reward = np.mean(BOLA_reward_all)
	RL_total_reward = np.mean(RL_reward_all)
	robustMPC_total_reward = np.mean(robustMPC_reward_all)
	fastMPC_total_reward = np.mean(fastMPC_reward_all)

	sns.set(style="ticks", font_scale=1.4)
	marker = itertools.cycle(("o", "v", "p", "d", "h", "s", "^"))
	linestyle = itertools.cycle(("-", "-.", "--", ":"))

	plt.xlim(0, len(BB_reward_all)+10)
	# plt.ylim(-75.0, 175.0)

	plt.plot(RL_reward_all, label="RL " + "{:.2f}".format(RL_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(BOLA_reward_all, label="BO " + "{:.2f}".format(BOLA_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(robustMPC_reward_all, label="rM " + "{:.2f}".format(robustMPC_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(FIXED_reward_all, label="FX " + "{:.2f}".format(FIXED_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(fastMPC_reward_all, label="fM " + "{:.2f}".format(fastMPC_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(BB_reward_all, label="BB " + "{:.2f}".format(BB_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(FESTIVE_reward_all, label="FV " + "{:.2f}".format(FESTIVE_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	plt.plot(RB_reward_all, label="RB " + "{:.2f}".format(RB_total_reward),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	plt.legend()
	plt.tight_layout()
	plt.savefig("hab_rewards_abr.pdf", format="pdf")
	plt.savefig("hab_rewards_abr.png", format="png")
	plt.show()
	plt.clf()
	


if __name__ == '__main__':
	main()