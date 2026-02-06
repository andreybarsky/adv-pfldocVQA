import pickle
import editdistance

DOA = 1

# filename = 'results_DonutModel.pkl'
# path = rf"/home-local/mpintore/recovery_2109/adv_docVQA/resultsACCUMULATION/Exp2/5-bottom_right_corner-DonutModel/{filename}"
# filename = 'results_Pix2StructModel.pkl'
# path = rf"/home-local/mpintore/aug_exp/adv_docVQA/resultsACCUMULATION/Exp2/5-bottom_right_corner-Pix2StructModel/{filename}"

# doa path
filename = 'results_Pix2StructModel.pkl'
path = rf"/home-local/mpintore/aug_exp/adv_docVQA/resultsACCUMULATION/Exp3/5-bottom_right_corner-Pix2StructModel/{filename}"


class Evaluator:
    def __init__(self, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        self.total_accuracies = []
        self.total_anls = []

        self.best_accuracy = 0
        # self.best_anls = 0
        self.best_epoch = 0

    def get_metrics(self, gt_answers, preds, answer_types=None, update_global_metrics=True):
        answer_types = answer_types if answer_types is not None else ['string' for batch_idx in range(len(gt_answers))]
        batch_accuracy = []
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])

            batch_accuracy.append(self._calculate_accuracy(gt, pred, answer_types[batch_idx]))
            batch_anls.append(self._calculate_anls(gt, pred, answer_types[batch_idx]))

        # if accumulate_metrics:
        #     self.total_accuracies.extend(batch_accuracy)
        #     self.total_anls.extend(batch_anls)

        return {'accuracy': batch_accuracy, 'anls': batch_anls}

    def get_retrieval_metric(self, gt_answer_page, pred_answer_page):
        retrieval_precision = [1 if gt == pred else 0 for gt, pred in zip(gt_answer_page, pred_answer_page)]
        return retrieval_precision

    def update_global_metrics(self, accuracy, anls, current_epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = current_epoch
            return True

        else:
            return False

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()

    def _calculate_accuracy(self, gt, pred, answer_type):

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        for gt_elm in gt:
            if gt_elm == pred:
                return 1

        return 0

    def _calculate_anls(self, gt, pred, answer_type):
        if len(pred) == 0:
            return 0

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls
    
def report(results):
    evaluator = Evaluator(case_sensitive=False)

    asr_successful_batches = 0
    total_anls_gt = []
    total_anls_tgt = []
    
    total_batches_processed = 0
    total_samples_processed = 0

    if not results:
        return

    print('*********** Starting report ***********')
    
    for item_key, value in results.items():
        
        try:
            batch_size = len(value['questions'])
            if not all(len(v) == batch_size for k, v in value.items()):
                 print(f"\nERROR {item_key} len mismatch ")
                 continue
            
            total_batches_processed += 1

        except Exception as e:
            print(f"\nERROR - {item_key}")
            continue

        all_samples_in_batch_match = True
        samples_in_batch_processed = 0

        for i in range(batch_size): # for each element in the batch
            try:
                y_pred_str = value['y_pred'][i]
                y_pred_adv_str = value['y_pred_adv'][i]
                if not DOA:
                    target_str = value['target'][i]
                gt_set = value['gt'][i]
                gt_list = list(gt_set)
                
                total_samples_processed += 1
                samples_in_batch_processed += 1

            except Exception as e:
                print(f"\nERROR: something wrong index {i} in batch '{item_key}'")
                all_samples_in_batch_match = False 
                continue 

            
            # ASR logic -> if an answer is wrong, the whole batch counts as ASR = 0
            # We don't normalize the answer for this kind of metric
            if (DOA and y_pred_str == y_pred_adv_str) or (not DOA and y_pred_adv_str != target_str):
                all_samples_in_batch_match = False

            pred_adv_norm = evaluator._preprocess_str(y_pred_adv_str)
            if not DOA:
                target_norm = evaluator._preprocess_str(target_str)

            gt_norm_list = [evaluator._preprocess_str(g) for g in gt_list]  
            anls_gt_score = evaluator._calculate_anls(gt_norm_list, pred_adv_norm, answer_type="string")
            total_anls_gt.append(anls_gt_score)

            if not DOA:
                target_list_norm = [target_norm] 
                anls_tgt_score = evaluator._calculate_anls(target_list_norm, pred_adv_norm, answer_type="string")
                total_anls_tgt.append(anls_tgt_score)


        # increment asr only if all the samples in the batch match the target answer
        if all_samples_in_batch_match and samples_in_batch_processed == batch_size:
            asr_successful_batches += 1

    # finally, compute all the metrics
    if total_batches_processed == 0:
        print("*********** Report ***********")
        print("******************************")
        return

    # ASR is computed on batches
    final_asr = (asr_successful_batches / total_batches_processed) * 100

    # ANLS is computed on the total number of samples
    if total_samples_processed == 0:
        final_anls_gt = 0.0
        final_anls_tgt = 0.0
    else:
        final_anls_gt = (sum(total_anls_gt) / len(total_anls_gt)) * 100 if total_anls_gt else 0.0
        final_anls_tgt = (sum(total_anls_tgt) / len(total_anls_tgt)) * 100 if total_anls_tgt else 0.0

    print('*********** Report ***********')
    print(f'# Number of valid batches (for ASR) = {total_batches_processed}')
    print(f'# Number of valid samples (for ANLS) = {total_samples_processed}')
    print('---')
    print(f'# ASR (Batch-level) = {final_asr:.2f}%')
    print(f"# ANLS_GT (Sample-level): {final_anls_gt:.2f}%")
    print(f"# ANLS_B (Sample-level): {final_anls_tgt:.2f}%")
    print('******************************')


with open(path, "rb") as f:
    data = pickle.load(f)
    report(data)
    print(path)