import logging
import os
import torch
import torch.utils.data as data
from collections import defaultdict
import json


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, word2index, args):
        """Reads source and target sequences from txt files."""
        self.word2index = word2index
        self.data = data
        self.args = args
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["context_text_lu"] = self.data["context_lu"][index]
        item["context_text_lsu"] = self.data["context_lsu"][index]  
        item["context_text_lsup"] = self.data["context_lsup"][index]         
        item["target_text"] = self.data["target"][index]
        item["target_text_lsu"] = self.data["target_lsu"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["yon_text"] = self.data["yon"][index]

        inputs_lu = self.preprocess([self.data["context_lu"][index],
                                  self.data["vads_lu"][index],
                                  self.data["vad_lu"][index],
                                  self.data["concepts_lu"][index]])

        inputs = self.preprocess([self.data["context"][index],
                                  self.data["vads"][index],
                                  self.data["vad"][index],
                                  self.data["concepts"][index]])

        inputs_lsu = self.preprocess([self.data["context_lsu"][index],
                                  self.data["vads_lsu"][index],
                                  self.data["vad_lsu"][index],
                                  self.data["concepts_lsu"][index]])  

        inputs_lsup = self.preprocess([self.data["context_lsup"][index],
                                  self.data["vads_lsup"][index],
                                  self.data["vad_lsup"][index],
                                  self.data["concepts_lsup"][index]]) 
              

        item["context_lu"], item["context_ext_lu"], item["context_mask_lu"], item["vads_lu"], item["vad_lu"], \
        item["concept_text_lu"], item["concept_lu"], item["concept_ext_lu"], item["concept_vads_lu"], item["concept_vad_lu"], \
        item["oovs_lu"]= inputs_lu

        item["context"], item["context_ext"], item["context_mask"], item["vads"], item["vad"], \
        item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], item["concept_vad"], \
        item["oovs"]= inputs

        item["context_lsu"], item["context_ext_lsu"], item["context_mask_lsu"], item["vads_lsu"], item["vad_lsu"], \
        item["concept_text_lsu"], item["concept_lsu"], item["concept_ext_lsu"], item["concept_vads_lsu"], item["concept_vad_lsu"], \
        item["oovs_lsu"]= inputs_lsu
        
        item["context_lsup"], item["context_ext_lsup"], item["context_mask_lsup"], item["vads_lsup"], item["vad_lsup"], \
        item["concept_text_lsup"], item["concept_lsup"], item["concept_ext_lsup"], item["concept_vads_lsup"], item["concept_vad_lsup"], \
        item["oovs_lsup"]= inputs_lsup

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_lsu"] = self.preprocess(item["target_text_lsu"], anw=True)
        item["target_ext"] = self.target_oovs(item["target_text"], item["oovs"])
        item["target_ext_lsu"] = self.target_oovs(item["target_text_lsu"], item["oovs_lsu"])
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"],
                                                                     self.emo_map)  
        item["emotion_widx"] = self.word2index[item["emotion_text"]]

        return item

    def __len__(self):
        return len(self.data["target"])

    def target_oovs(self, target, oovs):
        ids = []
        for w in target:
            if w not in self.word2index:
                if w in oovs:
                    ids.append(len(self.word2index) + oovs.index(w))
                else:
                    ids.append(self.args.UNK_idx)
            else:
                ids.append(self.word2index[w])
        ids.append(self.args.EOS_idx)
        return torch.LongTensor(ids)

    def process_oov(self, context, concept):  
        ids = []
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.word2index:
                    i = self.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.word2index) + oov_num)

        for sentence_concept in concept:
            for token_concept in sentence_concept:
                for c in token_concept:
                    if c not in oovs and c not in self.word2index:
                        oovs.append(c)
        return ids, oovs

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if anw:
            sequence = [self.word2index[word] if word in self.word2index else self.args.UNK_idx for word in arr] + [self.args.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            context = arr[0]
            context_vads = arr[1]
            context_vad = arr[2]
            concept = [arr[3][l][0] for l in range(len(arr[3]))]
            concept_vads = [arr[3][l][1] for l in range(len(arr[3]))]
            concept_vad = [arr[3][l][2] for l in range(len(arr[3]))]

            X_dial = [self.args.CLS_idx]
            X_dial_ext = [self.args.CLS_idx]
            X_mask = [self.args.CLS_idx] 
            X_vads = [[0.5, 0.0, 0.5]]
            X_vad = [0.0]

            X_concept_text = defaultdict(list)
            X_concept = [[]] 
            X_concept_ext = [[]]
            X_concept_vads = [[0.5, 0.0, 0.5]]
            X_concept_vad = [0.0]
            assert len(context) == len(concept)

            X_ext, X_oovs = self.process_oov(context, concept)
            X_dial_ext += X_ext

            for i, sentence in enumerate(context):
                X_dial += [self.word2index[word] if word in self.word2index else self.args.UNK_idx for word in sentence]
                spk = self.word2index["[USR]"] if i % 2 == 0 else self.word2index["[SYS]"]
                X_mask += [spk for _ in range(len(sentence))]
                X_vads += context_vads[i]
                X_vad += context_vad[i]

                for j, token_conlist in enumerate(concept[i]):
                    if token_conlist == []:
                        X_concept.append([])
                        X_concept_ext.append([])
                        X_concept_vads.append([0.5, 0.0, 0.5]) 
                        X_concept_vad.append(0.0)
                    else:
                        X_concept_text[sentence[j]] += token_conlist[:self.args.concept_num]
                        X_concept.append([self.word2index[con_word] if con_word in self.word2index else self.args.UNK_idx for con_word in token_conlist[:self.args.concept_num]])

                        con_ext = []
                        for con_word in token_conlist[:self.args.concept_num]:
                            if con_word in self.word2index:
                                con_ext.append(self.word2index[con_word])
                            else:
                                if con_word in X_oovs:
                                    con_ext.append(X_oovs.index(con_word) + len(self.word2index))
                                else:
                                    con_ext.append(self.args.UNK_idx)
                        X_concept_ext.append(con_ext)
                        X_concept_vads.append(concept_vads[i][j][:self.args.concept_num])
                        X_concept_vad.append(concept_vad[i][j][:self.args.concept_num])

                        assert len([self.word2index[con_word] if con_word in self.word2index else self.args.UNK_idx for con_word in token_conlist[:self.args.concept_num]]) == len(concept_vads[i][j][:self.args.concept_num]) == len(concept_vad[i][j][:self.args.concept_num])
            assert len(X_dial) == len(X_mask) == len(X_concept) == len(X_concept_vad) == len(X_concept_vads)

            return X_dial, X_dial_ext, X_mask, X_vads, X_vad, \
                   X_concept_text, X_concept, X_concept_ext, X_concept_vads, X_concept_vad, \
                   X_oovs

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion] 


    def collate_fn(self, batch_data):
        def merge(sequences): 
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
            return padded_seqs, lengths

        def merge_concept(samples, samples_ext, samples_vads, samples_vad):
            concept_lengths = []  
            token_concept_lengths = [] 
            concepts_list = []
            concepts_ext_list = []
            concepts_vads_list = []
            concepts_vad_list = []

            for i, sample in enumerate(samples):
                length = 0 
                sample_concepts = []
                sample_concepts_ext = []
                token_length = []
                vads = []
                vad = []

                for c, token in enumerate(sample):
                    if token == []:
                        token_length.append(0)
                        continue
                    length += len(token)
                    token_length.append(len(token))
                    sample_concepts += token
                    sample_concepts_ext += samples_ext[i][c]
                    vads += samples_vads[i][c]
                    vad += samples_vad[i][c]

                if length > self.args.total_concept_num:
                    value, rank = torch.topk(torch.LongTensor(vad), k=self.args.total_concept_num)

                    new_length = 1
                    new_sample_concepts = [self.args.SEP_idx]
                    new_sample_concepts_ext = [self.args.SEP_idx]
                    new_token_length = []
                    new_vads = [[0.5,0.0,0.5]]
                    new_vad = [0.0]

                    cur_idx = 0
                    for ti, token in enumerate(sample):
                        if token == []:
                            new_token_length.append(0)
                            continue
                        top_length = 0
                        for ci, con in enumerate(token):
                            point_idx = cur_idx + ci
                            if point_idx in rank:
                                top_length += 1
                                new_length += 1
                                new_sample_concepts.append(con)
                                new_sample_concepts_ext.append(samples_ext[i][ti][ci])
                                new_vads.append(samples_vads[i][ti][ci])
                                new_vad.append(samples_vad[i][ti][ci])
                                assert len(samples_vads[i][ti][ci]) == 3

                        new_token_length.append(top_length)
                        cur_idx += len(token)

                    new_length += 1 
                    new_sample_concepts = [self.args.SEP_idx] + new_sample_concepts
                    new_sample_concepts_ext = [self.args.SEP_idx] + new_sample_concepts_ext
                    new_vads = [[0.5,0.0,0.5]] + new_vads
                    new_vad = [0.0] + new_vad

                    concept_lengths.append(new_length) 
                    token_concept_lengths.append(new_token_length) 
                    concepts_list.append(new_sample_concepts)
                    concepts_ext_list.append(new_sample_concepts_ext)
                    concepts_vads_list.append(new_vads)
                    concepts_vad_list.append(new_vad)
                    assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(new_sample_concepts_ext), "The number of concept tokens, vads [*,*,*], and vad * should be the same."
                    assert len(new_token_length) == len(token_length)
                else:
                    length += 1
                    sample_concepts = [self.args.SEP_idx] + sample_concepts
                    sample_concepts_ext = [self.args.SEP_idx] + sample_concepts_ext
                    vads = [[0.5,0.0,0.5]] + vads
                    vad = [0.0] + vad

                    concept_lengths.append(length)
                    token_concept_lengths.append(token_length)
                    concepts_list.append(sample_concepts)
                    concepts_ext_list.append(sample_concepts_ext)
                    concepts_vads_list.append(vads)
                    concepts_vad_list.append(vad)

            if max(concept_lengths) != 0:
                padded_concepts = torch.ones(len(samples), max(concept_lengths)).long() 
                padded_concepts_ext = torch.ones(len(samples), max(concept_lengths)).long() 
                padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(samples), max(concept_lengths), 1)
                padded_concepts_vad = torch.FloatTensor([[0.0]]).repeat(len(samples), max(concept_lengths)) 
                padded_mask = torch.ones(len(samples), max(concept_lengths)).long() 

                for j, concepts in enumerate(concepts_list):
                    end = concept_lengths[j]
                    if end == 0:
                        continue
                    padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                    padded_concepts_ext[j, :end] = torch.LongTensor(concepts_ext_list[j][:end])
                    padded_concepts_vads[j, :end, :] = torch.FloatTensor(concepts_vads_list[j][:end])
                    padded_concepts_vad[j, :end] = torch.FloatTensor(concepts_vad_list[j][:end])
                    padded_mask[j, :end] = self.args.KG_idx 

                return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
            else:  
                return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor([]), torch.Tensor([]), torch.Tensor([])

        def merge_vad(vads_sequences, vad_sequences):
            lengths = [len(seq) for seq in vad_sequences]
            padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(vads_sequences), max(lengths), 1)
            padding_vad = torch.FloatTensor([[0.5]]).repeat(len(vads_sequences), max(lengths))

            for i, vads in enumerate(vads_sequences):
                end = lengths[i] 
                padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
                padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
            return padding_vads, padding_vad 

        def adj_mask(context, context_lengths, concepts, token_concept_lengths):
            '''

            :param self:
            :param context: (bsz, max_context_len)
            :param context_lengths: [] len=bsz
            :param concepts: (bsz, max_concept_len)
            :param token_concept_lengths: [] len=bsz;
            :return:
            '''

            bsz, max_context_len = context.size()
            max_concept_len = concepts.size(1) 
            adjacency_size = max_context_len + max_concept_len
            adjacency = torch.ones(bsz, max_context_len, adjacency_size) 

            for i in range(bsz):

                adjacency[i, 0, :context_lengths[i]] = 0
                adjacency[i, :context_lengths[i], 0] = 0

                con_idx = max_context_len+1  
                for j in range(context_lengths[i]):
                    adjacency[i, j, j - 1] = 0 

                    token_concepts_length = token_concept_lengths[i][j]
                    if token_concepts_length == 0:
                        continue
                    else:
                        adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0
                        adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0
                        con_idx += token_concepts_length
            return adjacency

        batch_data.sort(key=lambda x: len(x["context"]), reverse=True)
        item_info = {}
        for key in batch_data[0].keys():
            item_info[key] = [d[key] for d in batch_data]

        assert len(item_info['context']) == len(item_info['vad'])

        context_batch_lu, context_lengths_lu = merge(item_info['context_lu'])
        context_ext_batch_lu, _ = merge(item_info['context_ext_lu'])
        mask_context_lu, _ = merge(item_info['context_mask_lu']) 


        context_batch, context_lengths = merge(item_info['context'])
        context_ext_batch, _ = merge(item_info['context_ext'])
        mask_context, _ = merge(item_info['context_mask']) 

        context_batch_lsu, context_lengths_lsu = merge(item_info['context_lsu'])
        context_ext_batch_lsu, _ = merge(item_info['context_ext_lsu'])
        mask_context_lsu, _ = merge(item_info['context_mask_lsu']) 

        context_batch_lsup, context_lengths_lsup = merge(item_info['context_lsup'])
        context_ext_batch_lsup, _ = merge(item_info['context_ext_lsup'])
        mask_context_lsup, _ = merge(item_info['context_mask_lsup'])

        context_vads_batch, context_vad_batch = merge_vad(item_info['vads'], item_info['vad'])
        context_vads_batch_lu, context_vad_batch_lu = merge_vad(item_info['vads_lu'], item_info['vad_lu'])
        context_vads_batch_lsu, context_vad_batch_lsu = merge_vad(item_info['vads_lsu'], item_info['vad_lsu'])        
        context_vads_batch_lsup, context_vad_batch_lsup = merge_vad(item_info['vads_lsup'], item_info['vad_lsup']) 

        assert context_batch.size(1) == context_vad_batch.size(1)

        concept_inputs_lu = merge_concept(item_info['concept_lu'],
                                       item_info['concept_ext_lu'],
                                       item_info["concept_vads_lu"],
                                       item_info["concept_vad_lu"]) 
        concept_batch_lu, concept_ext_batch_lu, concept_lengths_lu, mask_concept_lu, token_concept_lengths_lu, concepts_vads_batch_lu, concepts_vad_batch_lu = concept_inputs_lu

        concept_inputs = merge_concept(item_info['concept'],
                                       item_info['concept_ext'],
                                       item_info["concept_vads"],
                                       item_info["concept_vad"]) 
        concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths, concepts_vads_batch, concepts_vad_batch = concept_inputs

        concept_inputs_lsu = merge_concept(item_info['concept_lsu'],
                                       item_info['concept_ext_lsu'],
                                       item_info["concept_vads_lsu"],
                                       item_info["concept_vad_lsu"]) 
        concept_batch_lsu, concept_ext_batch_lsu, concept_lengths_lsu, mask_concept_lsu, token_concept_lengths_lsu, concepts_vads_batch_lsu, concepts_vad_batch_lsu = concept_inputs_lsu
        
        concept_inputs_lsup = merge_concept(item_info['concept_lsup'],
                                       item_info['concept_ext_lsup'],
                                       item_info["concept_vads_lsup"],
                                       item_info["concept_vad_lsup"])
        concept_batch_lsup, concept_ext_batch_lsup, concept_lengths_lsup, mask_concept_lsup, token_concept_lengths_lsup, concepts_vads_batch_lsup, concepts_vad_batch_lsup = concept_inputs_lsup

        if concept_batch.size()[0] != 0:
            adjacency_mask_batch = adj_mask(context_batch, context_lengths, concept_batch, token_concept_lengths)
        else:
            adjacency_mask_batch = torch.Tensor([])

        if concept_batch_lu.size()[0] != 0:
            adjacency_mask_batch_lu = adj_mask(context_batch_lu, context_lengths_lu, concept_batch_lu, token_concept_lengths_lu)
        else:
            adjacency_mask_batch_lu = torch.Tensor([])

        if concept_batch_lsu.size()[0] != 0:
            adjacency_mask_batch_lsu = adj_mask(context_batch_lsu, context_lengths_lsu, concept_batch_lsu, token_concept_lengths_lsu)
        else:
            adjacency_mask_batch_lsu = torch.Tensor([])

        if concept_batch_lsup.size()[0] != 0:
            adjacency_mask_batch_lsup = adj_mask(context_batch_lsup, context_lengths_lsup, concept_batch_lsup, token_concept_lengths_lsup)
        else:
            adjacency_mask_batch_lsup = torch.Tensor([])

        target_batch, target_lengths = merge(item_info['target'])
        target_ext_batch, _ = merge(item_info['target_ext'])

        target_batch_lsu, target_lengths_lsu = merge(item_info['target_lsu'])
        target_ext_batch_lsu, _ = merge(item_info['target_ext_lsu'])
        d = {}

        d["context_batch_lu"] = context_batch_lu.to(self.args.device) 
        d["context_batch_lu"] = context_batch_lu.to(self.args.device)
        d["context_ext_batch_lu"] = context_ext_batch_lu.to(self.args.device) 
        d["context_lengths_lu"] = torch.LongTensor(context_lengths_lu).to(self.args.device) 
        d["mask_context_lu"] = mask_context_lu.to(self.args.device)
        d["context_vads_lu"] = context_vads_batch_lu.to(self.args.device) 
        d["context_vad_lu"] = context_vad_batch_lu.to(self.args.device) 

        d["context_batch"] = context_batch.to(self.args.device) 
        d["context_batch"] = context_batch.to(self.args.device)
        d["context_ext_batch"] = context_ext_batch.to(self.args.device) 
        d["context_lengths"] = torch.LongTensor(context_lengths).to(self.args.device) 
        d["mask_context"] = mask_context.to(self.args.device)
        d["context_vads"] = context_vads_batch.to(self.args.device) 
        d["context_vad"] = context_vad_batch.to(self.args.device) 

        d["context_batch_lsu"] = context_batch_lsu.to(self.args.device) 
        d["context_batch_lsu"] = context_batch_lsu.to(self.args.device)
        d["context_ext_batch_lsu"] = context_ext_batch_lsu.to(self.args.device) 
        d["context_lengths_lsu"] = torch.LongTensor(context_lengths_lsu).to(self.args.device) 
        d["mask_context_lsu"] = mask_context_lsu.to(self.args.device)
        d["context_vads_lsu"] = context_vads_batch_lsu.to(self.args.device) 
        d["context_vad_lsu"] = context_vad_batch_lsu.to(self.args.device)  

        d["context_batch_lsup"] = context_batch_lsup.to(self.args.device)  
        d["context_batch_lsup"] = context_batch_lsup.to(self.args.device)
        d["context_ext_batch_lsup"] = context_ext_batch_lsup.to(self.args.device) 
        d["context_lengths_lsup"] = torch.LongTensor(context_lengths_lsup).to(self.args.device)  
        d["mask_context_lsup"] = mask_context_lsup.to(self.args.device)
        d["context_vads_lsup"] = context_vads_batch_lsup.to(self.args.device) 
        d["context_vad_lsup"] = context_vad_batch_lsup.to(self.args.device) 

        d["concept_batch_lu"] = concept_batch_lu.to(self.args.device) 
        d["concept_ext_batch_lu"] = concept_ext_batch_lu.to(self.args.device) 
        d["concept_lengths_lu"] = torch.LongTensor(concept_lengths_lu).to(self.args.device) 
        d["mask_concept_lu"] = mask_concept_lu.to(self.args.device) 
        d["concept_vads_batch_lu"] = concepts_vads_batch_lu.to(self.args.device) 
        d["concept_vad_batch_lu"] = concepts_vad_batch_lu.to(self.args.device) 
        d["adjacency_mask_batch_lu"] = adjacency_mask_batch_lu.bool().to(self.args.device)



        d["concept_batch"] = concept_batch.to(self.args.device) 
        d["concept_ext_batch"] = concept_ext_batch.to(self.args.device) 
        d["concept_lengths"] = torch.LongTensor(concept_lengths).to(self.args.device) 
        d["mask_concept"] = mask_concept.to(self.args.device) 
        d["concept_vads_batch"] = concepts_vads_batch.to(self.args.device)  
        d["concept_vad_batch"] = concepts_vad_batch.to(self.args.device) 
        d["adjacency_mask_batch"] = adjacency_mask_batch.bool().to(self.args.device)

        d["concept_batch_lsu"] = concept_batch_lsu.to(self.args.device)  
        d["concept_ext_batch_lsu"] = concept_ext_batch_lsu.to(self.args.device)
        d["concept_lengths_lsu"] = torch.LongTensor(concept_lengths_lsu).to(self.args.device)
        d["mask_concept_lsu"] = mask_concept_lsu.to(self.args.device) 
        d["concept_vads_batch_lsu"] = concepts_vads_batch_lsu.to(self.args.device) 
        d["concept_vad_batch_lsu"] = concepts_vad_batch_lsu.to(self.args.device) 
        d["adjacency_mask_batch_lsu"] = adjacency_mask_batch_lsu.bool().to(self.args.device)


        d["concept_batch_lsup"] = concept_batch_lsup.to(self.args.device)
        d["concept_ext_batch_lsup"] = concept_ext_batch_lsup.to(self.args.device)
        d["concept_lengths_lsup"] = torch.LongTensor(concept_lengths_lsup).to(self.args.device)
        d["mask_concept_lsup"] = mask_concept_lsup.to(self.args.device) 
        d["concept_vads_batch_lsup"] = concepts_vads_batch_lsup.to(self.args.device) 
        d["concept_vad_batch_lsup"] = concepts_vad_batch_lsup.to(self.args.device) 
        d["adjacency_mask_batch_lsup"] = adjacency_mask_batch_lsup.bool().to(self.args.device)

        d["target_batch"] = target_batch.to(self.args.device) 
        d["target_ext_batch"] = target_ext_batch.to(self.args.device)
        d["target_lengths"] = torch.LongTensor(target_lengths).to(self.args.device) 

        d["target_batch_lsu"] = target_batch_lsu.to(self.args.device) 
        d["target_ext_batch_lsu"] = target_ext_batch_lsu.to(self.args.device)
        d["target_lengths_lsu"] = torch.LongTensor(target_lengths_lsu).to(self.args.device) 

        d["target_emotion"] = torch.LongTensor(item_info['emotion']).to(self.args.device)
        d["emotion_label"] = torch.LongTensor(item_info['emotion_label']).to(self.args.device) 
        d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx']).to(self.args.device)
        assert d["emotion_widx"].size() == d["emotion_label"].size()

        d["context_txt"] = item_info['context_text']
        d["context_txt_lu"] = item_info['context_text_lu']
        d["context_txt_lsu"] = item_info['context_text_lsu']  
        d["context_txt_lsup"] = item_info['context_text_lsup']  
        d["target_txt"] = item_info['target_text']
        d["target_txt_lsu"] = item_info['target_text_lsu']
        d["emotion_txt"] = item_info['emotion_text']
        d["concept_txt"] = item_info['concept_text']
        d["concept_txt_lu"] = item_info['concept_text_lu']
        d["concept_txt_lsu"] = item_info['concept_text_lsu']    
        d["concept_txt_lsup"] = item_info['concept_text_lsup']   
        d["oovs"] = item_info["oovs"]
        d["yon_batch"] = item_info["yon_text"]
        

        return d


def write_config(args):
    if not args.test:
        if not os.path.exists(os.path.join(args.save_path, 'result', args.model)):
            os.makedirs(os.path.join(args.save_path, 'result', args.model))
        with open(os.path.join(args.save_path, 'result', args.model, 'config.txt'),'w') as the_file:
            for k, v in args.__dict__.items():
                if "False" in str(v):
                    pass
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k,v))


def flatten(t):
    return [item for sublist in t for item in sublist]


def load_dataset(args):
    print('file: ', args.dataset)
    if os.path.exists(args.dataset):
        print("LOADING empathetic_dialogue")
        with open(args.dataset, 'r') as f:
            [data_tra, data_val, data_tst, vocab] = json.load(f)
        with open(args.dataset, 'r') as f:
            [data_tra1, data_val1, data_tst1, vocab1] = json.load(f)
        with open(args.dataset, 'r') as f:
            [data_tra2, data_val2, data_tst2, vocab2] = json.load(f)
        with open(args.dataset, 'r') as f:
            [data_tra3, data_val3, data_tst3, vocab2] = json.load(f)
    else:
        print("data file not exists !!")
    
    print('ybybl')
    
    data_tra['context_lu'] = data_tra['context']
    data_tra['context_lsu'] = data_tra1['context']
    data_tra['context_lsup'] = data_tra2['context']
    data_tra['target_lsu'] = data_tra3['target']
    data_tra['concepts_lu'] = data_tra['concepts']
    data_tra['concepts_lsu'] = data_tra1['concepts']
    data_tra['concepts_lsup'] = data_tra2['concepts']
    data_tra['vads_lu'] = data_tra['vads']
    data_tra['vads_lsu'] = data_tra1['vads']
    data_tra['vads_lsup'] = data_tra2['vads']
    data_tra['vad_lu'] = data_tra['vad']
    data_tra['vad_lsu'] = data_tra1['vad']
    data_tra['vad_lsup'] = data_tra2['vad']
    data_val['context_lu'] = data_val['context']
    data_val['context_lsu'] = data_val1['context']
    data_val['context_lsup'] = data_val2['context']
    data_val['target_lsu'] = data_val3['target']
    data_val['concepts_lu'] = data_val['concepts']
    data_val['concepts_lsu'] = data_val1['concepts']
    data_val['concepts_lsup'] = data_val2['concepts']
    data_val['vads_lu'] = data_val['vads']
    data_val['vads_lsu'] = data_val1['vads']
    data_val['vads_lsup'] = data_val2['vads']
    data_val['vad_lu'] = data_val['vad']
    data_val['vad_lsu'] = data_val1['vad']
    data_val['vad_lsup'] = data_val2['vad']
    data_tst['context_lu'] = data_tst['context']
    data_tst['context_lsu'] = data_tst1['context']
    data_tst['context_lsup'] = data_tst2['context']
    data_tst['target_lsu'] = data_tst3['target']
    data_tst['concepts_lu'] = data_tst['concepts']
    data_tst['concepts_lsu'] = data_tst1['concepts']
    data_tst['concepts_lsup'] = data_tst2['concepts']
    data_tst['vads_lu'] = data_tst['vads']
    data_tst['vads_lsu'] = data_tst1['vads']
    data_tst['vads_lsup'] = data_tst2['vads']
    data_tst['vad_lu'] = data_tst['vad']
    data_tst['vad_lsu'] = data_tst1['vad']
    data_tst['vad_lsup'] = data_tst2['vad']
    data_tra['yon'] = data_tra3['emotion']
    data_val['yon'] = data_val3['emotion']
    data_tst['yon'] = data_tst3['emotion']

    for i in range(len(data_tra['context'])):
        data_tra['context_lu'][i] = [data_tra['context'][i][-1]]        
        data_tra['concepts_lu'][i] = [data_tra['concepts'][i][-1]]      
        data_tra['vads_lu'][i] = [data_tra['vads'][i][-1]]
        data_tra['vad_lu'][i] = [data_tra['vad'][i][-1]] 
     

    for i in range(len(data_val['context'])):
        data_val['context_lu'][i] = [data_val['context'][i][-1]]
        data_val['concepts_lu'][i] = [data_val['concepts'][i][-1]]
        data_val['vads_lu'][i] = [data_val['vads'][i][-1]]
        data_val['vad_lu'][i] = [data_val['vad'][i][-1]]
        
    for i in range(len(data_tst['context'])):
        data_tst['context_lu'][i] = [data_tst['context'][i][-1]]
        data_tst['concepts_lu'][i] = [data_tst['concepts'][i][-1]]
        data_tst['vads_lu'][i] = [data_tst['vads'][i][-1]]
        data_tst['vad_lu'][i] = [data_tst['vad'][i][-1]]

    for i in range(len(data_tra1['context'])):     
        if len(data_tra1['context'][i]) > 1:
            data_tra['context_lsu'][i] = [data_tra1['context'][i][-2]]
            data_tra['context_lsup'][i] = data_tra2['context'][i][:-1]
            data_tra['target_lsu'][i] = data_tra3['context'][i][-1]
            data_tra['concepts_lsu'][i] = [data_tra1['concepts'][i][-2]]
            data_tra['concepts_lsup'][i] = data_tra2['concepts'][i][:-1]
            data_tra['vads_lsu'][i] = [data_tra1['vads'][i][-2]]
            data_tra['vads_lsup'][i] = data_tra2['vads'][i][:-1]
            data_tra['vad_lsu'][i] = [data_tra1['vad'][i][-2]] 
            data_tra['vad_lsup'][i] = data_tra2['vad'][i][:-1]
            data_tra['yon'][i] = 1

        else:
            data_tra['yon'][i] = 0


    for i in range(len(data_tst1['context'])):     
        if len(data_tst1['context'][i]) > 1:
            data_tst['context_lsu'][i] = [data_tst1['context'][i][-2]]
            data_tst['context_lsup'][i] = data_tst2['context'][i][:-1]
            data_tst['target_lsu'][i] = data_tst3['context'][i][-1]
            data_tst['concepts_lsu'][i] = [data_tst1['concepts'][i][-2]]
            data_tst['concepts_lsup'][i] = data_tst2['concepts'][i][:-1]
            data_tst['vads_lsu'][i] = [data_tst1['vads'][i][-2]]
            data_tst['vads_lsup'][i] = data_tst2['vads'][i][:-1]
            data_tst['vad_lsu'][i] = [data_tst1['vad'][i][-2]] 
            data_tst['vad_lsup'][i] = data_tst2['vad'][i][:-1]
            data_tst['yon'][i] = 1

        else:
            data_tst['yon'][i] = 0      


    for i in range(len(data_val1['context'])):     
        if len(data_val1['context'][i]) > 1:
            data_val['context_lsu'][i] = [data_val1['context'][i][-2]]
            data_val['context_lsup'][i] = data_val2['context'][i][:-1]
            data_val['target_lsu'][i] = data_val3['context'][i][-1]
            data_val['concepts_lsu'][i] = [data_val1['concepts'][i][-2]]
            data_val['concepts_lsup'][i] = data_val2['concepts'][i][:-1]
            data_val['vads_lsu'][i] = [data_val1['vads'][i][-2]]
            data_val['vads_lsup'][i] = data_val2['vads'][i][:-1]
            data_val['vad_lsu'][i] = [data_val1['vad'][i][-2]] 
            data_val['vad_lsup'][i] = data_val2['vad'][i][:-1]
            data_val['yon'][i] = 1

        else:
            data_val['yon'][i] = 0
              

    if os.path.exists(args.dataset):
        print("LOADING empathetic_dialogue")
        with open(args.dataset, 'r') as f:
            [a, b, c, d] = json.load(f)
    data_tra['context'] = a['context']
    data_tra['concepts'] = a['concepts']
    data_tra['vads'] = a['vads']
    data_tra['vad'] = a['vad']
    data_val['context'] = b ['context']
    data_val['concepts'] = b ['concepts']
    data_val['vads'] = b ['vads']
    data_val['vad'] = b ['vad']   
    data_tst['context'] = c ['context']
    data_tst['concepts'] = c ['concepts']
    data_tst['vads'] = c ['vads']
    data_tst['vad'] = c ['vad']

    for i in range(3):

        print('sdsdsds')
        print(len(data_tra['context'][i]))
        print('yon')
        print(data_tra['yon'][i])

        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[concept of context]:')
        print(data_tra['concepts'][i])
        for si, sc in enumerate(data_tra['concepts'][i]):
            print('concept of sentence {} : {}'.format(si, flatten(sc[0])))
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")

        print('sdsdsds_lu')
        print(len(data_tra['context_lu'][i]))

        print('[emotion]:', data_tra['emotion'][i])
        print('[contaext]:', [' '.join(u) for u in data_tra['context_lu'][i]])
        print('[concept of context]:')
        for si, sc in enumerate(data_tra['concepts_lu'][i]):
            print('concept of sentence {} : {}'.format(si, flatten(sc[0])))
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")

        print('sdsdsds_lsu')
        print(len(data_tra['context_lsu'][i]))

        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context_lsu'][i]])
        print('[concept of context]:')
        print(data_tra['concepts_lsu'][i])
        for si, sc in enumerate(data_tra['concepts_lsu'][i]):
            print('concept of sentence {} : {}'.format(si, flatten(sc[0])))
        print('[target]:', ' '.join(data_tra['target_lsu'][i]))
        print(" ")

        print('sdsdsds_lsup')
        print(len(data_tra['context_lsup'][i]))

        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context_lsup'][i]])
        print('[concept of context]:')
        for si, sc in enumerate(data_tra['concepts_lsup'][i]):
            print('concept of sentence {} : {}'.format(si, flatten(sc[0])))
        print('[target]:', ' '.join(data_tra['target_lsu'][i]))
        print(" ")

    print("train length: ", len(data_tra['situation']))
    print("valid length: ", len(data_val['situation']))
    print("test length: ", len(data_tst['situation']))

    return data_tra, data_val, data_tst, vocab


def prepare_data_seq(args, batch_size=16):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset(args) 
    word2index, word2count, index2word, n_words = vocab

    logging.info("Vocab  {} ".format(n_words))

    dataset_train = Dataset(pairs_tra, word2index, args)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=dataset_train.collate_fn)

    dataset_valid = Dataset(pairs_val, word2index, args)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=dataset_valid.collate_fn)

    dataset_test = Dataset(pairs_tst, word2index, args)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=dataset_test.collate_fn)
    
 
    write_config(args)
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)