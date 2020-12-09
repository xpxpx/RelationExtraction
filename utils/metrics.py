from collections import Counter


class Metrics:
    def __init__(self, ignore_index=''):
        self.ignore_index = ignore_index

    def compute(self, total_pred, total_gold):
        correct_by_relation = Counter()
        pred_by_relation = Counter()
        gold_by_relation = Counter()
        for pred, gold in zip(total_pred, total_gold):
            if gold == self.ignore_index and pred == self.ignore_index:
                pass
            elif gold == self.ignore_index and pred != self.ignore_index:
                pred_by_relation[pred] += 1
            elif gold != self.ignore_index and pred == self.ignore_index:
                gold_by_relation[gold] += 1
            else:
                pred_by_relation[pred] += 1
                gold_by_relation[gold] += 1
                if gold == pred:
                    correct_by_relation[pred] += 1

        # compute micro_f1
        micro_precision = 0.0
        if sum(pred_by_relation.values()) > 0:
            micro_precision = sum(correct_by_relation.values()) / sum(pred_by_relation.values())

        micro_recall = 0.0
        if sum(gold_by_relation.values()) > 0:
            micro_recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())

        micro_f1 = 0.0
        if (micro_precision + micro_recall) > 0.0:
            micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)

        # compute macro_f1
        total_precision = []
        total_recall = []
        total_f1 = []

        total_relation = set(pred_by_relation.keys()) | set(gold_by_relation.keys())
        for relation in total_relation:
            precision = 0.0
            if pred_by_relation[relation] > 0:
                precision = correct_by_relation[relation] / pred_by_relation[relation]

            recall = 0.0
            if gold_by_relation[relation] > 0:
                recall = correct_by_relation[relation] / gold_by_relation[relation]

            f1 = 0.0
            if (precision + recall) > 0.0:
                f1 = 2.0 * precision * recall / (precision + recall)

            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(f1)

        macro_precision = sum(total_precision) / len(total_precision)
        macro_recall = sum(total_recall) / len(total_recall)
        macro_f1 = sum(total_f1) / len(total_f1)

        return {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
