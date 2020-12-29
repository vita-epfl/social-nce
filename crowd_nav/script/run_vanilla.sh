# --------------------------------------

export CPU_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0

# --------------------------------------

DATA_PERCENT=0.50
NUM_EPOCH=200

PATIENCE=$((NUM_EPOCH/10))

PREFIX=data/output/imitate-baseline-data-${DATA_PERCENT}
FILELOG=${PREFIX}/test_log.txt
FILERES=${PREFIX}/test_result.txt

# --------------------------------------

taskset -c $CPU_DEVICES python imitate.py --contrast_weight=0.0 --gpu --percent_label=${DATA_PERCENT} --num_epoch=${NUM_EPOCH} --scheduler_patience=${PATIENCE}

# --------------------------------------

date |& tee ${FILELOG}
for (( e=4; e<=${NUM_EPOCH}; e=e+5 )); do
	CKPTNAME=$(printf "%02d" $e)
	taskset -c $CPU_DEVICES python test.py --policy='sail' --circle --model_file=${PREFIX}/policy_net_${CKPTNAME}.pth |& tee -a ${FILELOG}
done

# --------------------------------------

echo "dataset, epoch, success, collision, time, reward, variance" |& tee ${FILERES}
cat ${FILELOG} \
	| grep "Loaded policy from\|TEST   success:" \
	| sed "s/.*-data-\([0-9.]*\)*.policy_net_\([0-9]*\).*/\1, \2,/g" \
	| sed "s/.*INFO: TEST   success: \([0-9.]*\), collision: \([0-9.]*\), nav time: \([0-9.]*\), reward: \([0-9.-]*\) +- \([0-9.]*\)/\1, \2, \3, \4, \5/g" \
	| paste -d " " - - \
	|& tee -a ${FILERES}
