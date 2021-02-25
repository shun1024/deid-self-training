from pathlib import Path
from typing import Union, List
import pickle, time, logging

import flair.nn
from flair.data import Token
from flair.datasets import DataLoader
from flair.training_utils import (
    log_line,
    add_file_handler,
    Result,
    Metric,
    store_embeddings,
)

import torch
from torch.utils.tensorboard import SummaryWriter

from augmentations import augment
from utils import save_to_ckpt

log = logging.getLogger("flair")


def evaluate(
    model,
    data_loader: DataLoader,
    out_path: Path = None,
) -> (Result, float):
    if type(out_path) == str:
        out_path = Path(out_path)

    with torch.no_grad():
        eval_loss = 0

        batch_no: int = 0

        metric = Metric("Evaluation", beta=model.beta)
        diff_metric = Metric("Evaluation", beta=model.beta)

        lines: List[str] = []

        if model.use_crf:
            transitions = model.transitions.detach().cpu().numpy()
        else:
            transitions = None

        for batch in data_loader:
            batch_no += 1

            with torch.no_grad():
                features = model.forward(batch)
                loss = model._calculate_loss(features, batch)
                tags, _ = model._obtain_labels(
                    feature=features,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=False,
                )

            eval_loss += loss

            for (sentence, sent_tags) in zip(batch, tags):
                for (token, tag) in zip(sentence.tokens, sent_tags):
                    token: Token = token
                    token.add_tag("predicted", tag.value, tag.score)

                    # append both to file for evaluation
                    eval_line = "{} {} {} {}\n".format(
                        token.text,
                        token.get_tag(model.tag_type).value,
                        tag.value,
                        tag.score,
                    )
                    lines.append(eval_line)
                lines.append("\n")

            def add_tags(spans, tag_names, new_name):
                new_tags = []
                for tag in spans:
                    if tag.tag in tag_names:
                        new_tags.append((new_name, tag.text))
                return new_tags

            def add_to_metric(metric, gold_tags, predicted_tags):
                # check for true positives, false positives and false negatives
                for tag, prediction in predicted_tags:
                    if (tag, prediction) in gold_tags:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        metric.add_fn(tag)
                    else:
                        metric.add_tn(tag)
                return metric

            is_golden_mimic = False
            is_predicted_mimic = False
            for sentence in batch:
                for tag in sentence.get_spans(model.tag_type):
                    if tag.tag == "NAME":
                        is_golden_mimic = True
                        break

            for sentence in batch:
                for tag in sentence.get_spans("predicted"):
                    if tag.tag == "NAME":
                        is_predicted_mimic = True
                        break

            for sentence in batch:
                gold_tags = [(tag.tag, tag.text) for tag in sentence.get_spans(model.tag_type)]
                predicted_tags = [(tag.tag, tag.text) for tag in sentence.get_spans("predicted")]
                metric = add_to_metric(metric, gold_tags, predicted_tags)

                if not is_golden_mimic:
                    gold_tags = add_tags(sentence.get_spans(model.tag_type), ['PATIENT', 'DOCTOR'], 'NAME')

                if not is_predicted_mimic:
                    predicted_tags = add_tags(sentence.get_spans("predicted"), ['PATIENT', 'DOCTOR'], 'NAME')

                diff_metric = add_to_metric(diff_metric, gold_tags, predicted_tags)

        eval_loss /= batch_no

        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        detailed_result = (
            f"\nMICRO_AVG: acc {metric.micro_avg_accuracy():.4f} - f1-score {metric.micro_avg_f_score():.4f}"
            f"\nMACRO_AVG: acc {metric.macro_avg_accuracy():.4f} - f1-score {metric.macro_avg_f_score():.4f}"
        )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )
        return result, eval_loss, diff_metric.f_score('NAME')


class CustomTrainer(flair.trainers.ModelTrainer):

    def dev_step(self, corpus, name, writer, mini_batch_size=32):
        # evaluate on train / dev / test split depending on training settings
        self.model.eval()
        eval_result, dev_loss, name_f1 = evaluate(
            self.model,
            DataLoader(corpus, batch_size=mini_batch_size, num_workers=8),
            embedding_storage_mode="none",
        )
        log.info(f"DEV {name} : loss {dev_loss} - score {eval_result.main_score}")
        log.info(f"DEV {name} : NAME F1 {name_f1}")

        if 'out' in name:
            current_score = name_f1
        else:
            current_score = eval_result.main_score

        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
        store_embeddings(corpus, "none")
        if self.use_tensorboard:
            writer.add_scalar("%s/micro_f1" % name, eval_result.main_score, self.epoch)
            writer.add_scalar("%s/name_micro_f1" % name, name_f1, self.epoch)
            writer.add_scalar("%s/loss" % name, dev_loss, self.epoch)
        return current_score

    def cutsom_train(
        self,
        base_path: Union[Path, str],
        unlabel_data: flair.datasets.ColumnDataset,
        out_corpus: flair.data.Corpus = None,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 100,
        train_step_ratio: int = 5,
        min_learning_rate: float = 0.0001,
        unlabel_batch_ratio: int = 0,
        unlabel_weight: float = 0,
        augment_method: str = 'word_replace',
        augment_prob: float = 0.15,
        temperature: float = 1,
        train_from_scratch: bool = False,
        **kwargs,
    ) -> dict:

        if self.use_tensorboard:
            writer = SummaryWriter(base_path)

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")
        optimizer: torch.optim.Optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)

        if unlabel_batch_ratio > 0:
            teacher = pickle.loads(pickle.dumps(self.model))

        if train_from_scratch:
            log.info('Resetting the model')
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.epoch = 0
            train_data = self.corpus.train

            while self.epoch < max_epochs:
                # get new learning rate
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                # stop training if learning rate becomes too small
                if learning_rate < min_learning_rate:
                    break

                if self.epoch % train_step_ratio == 0:
                    # validation steps
                    self.model.eval()
                    log.info("Saving model & corpus to local directory")
                    save_to_ckpt(base_path, self.model, self.corpus, unlabel_data)

                # training steps
                self.epoch += 1
                self.model.train()
                batch_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True, num_workers=8)

                if unlabel_batch_ratio > 0:
                    unlabel_batch_size = mini_batch_size * unlabel_batch_ratio
                    unlabel_batch_loader = DataLoader(unlabel_data, batch_size=unlabel_batch_size, shuffle=True,
                                                      num_workers=8)

                train_loss = 0
                unsup_train_loss = 0

                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))

                # process mini-batches
                batch_time = 0
                for batch_no, batch in enumerate(batch_loader):
                    start_time = time.time()

                    # zero the gradients on the model and optimizer
                    torch.cuda.empty_cache()
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # forward pass
                    loss = self.model._calculate_loss(self.model.forward(batch), batch)
                    loss.backward()

                    # teacher loss
                    if unlabel_batch_ratio > 0:
                        # calculate the unlabel loss
                        # if necessary, make batch_steps
                        unbatch_ori_batch = next(iter(unlabel_batch_loader))
                        teacher_output = teacher.forward(unbatch_ori_batch).detach()
                        teacher_prob = torch.nn.functional.softmax(teacher_output / temperature, -1)

                        unbatch_aug_batch = augment(unbatch_ori_batch, augment_method, augment_prob)
                        student_output = self.model.forward(unbatch_aug_batch)

                        # forward pass
                        unlabel_loss = unlabel_weight * torch.mean(
                            torch.sum(- teacher_prob * torch.nn.functional.log_softmax(student_output, -1), -1))
                        unlabel_loss.backward()

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_batches += 1
                    train_loss += loss.item()
                    if unlabel_batch_ratio > 0:
                        unsup_train_loss += unlabel_loss.item()
                        store_embeddings(unbatch_ori_batch, "none")

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(batch, "none")
                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        log.info(
                            f"epoch {self.epoch} - iter {seen_batches}/{total_number_of_batches} - loss "
                            f"{train_loss / seen_batches:.4f} - unlabel_loss {unsup_train_loss / seen_batches:.4f}"
                            f" - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                        )
                        batch_time = 0

                train_loss /= seen_batches
                if unlabel_batch_ratio > 0:
                    unsup_train_loss /= seen_batches

                log_line(log)
                log.info(
                    f"EPOCH {self.epoch} done: loss {train_loss:.4f} - unsup {unsup_train_loss:.4f} "
                    f"- lr {learning_rate:.4f}"
                )

                if self.use_tensorboard:
                    writer.add_scalar("train/loss", train_loss, self.epoch)
                    writer.add_scalar("train/learning_rate", learning_rate, self.epoch)
                    if unlabel_batch_ratio > 0:
                        writer.add_scalar("train/unsup_train_loss", unsup_train_loss, self.epoch)

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if self.use_tensorboard:
                writer.close()

        # test best model if test data is present
        final_score = self.final_test(base_path, mini_batch_size, 8)
        log.removeHandler(log_handler)

        if self.use_tensorboard:
            writer.add_scalar("test/score", final_score)
            writer.close()

        return {
            "test_score": final_score
        }
