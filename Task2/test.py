y_score = enc_probs[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
roc_auc = auc(fpr, tpr)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("ROC-кривая и Precision-Recall-кривая")
lw = 2
ax1.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
ax1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend()

precision, recall, thresh = precision_recall_curve(y_test, y_score)
pr_auc = auc(recall, precision)
ax2.step(recall, precision, color='b',
         where='post', label='PR curve (area = %0.2f)' % pr_auc)
ax2.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_ylim([0.0, 1.05])
ax2.set_xlim([0.0, 1.0])
ax2.legend()
