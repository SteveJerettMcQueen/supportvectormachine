import email

import charac_feats as cf
import word_feats as wf
import syntac_feats  as syf
import struct_feats as stf
import funct_word_feats as fwf

from nltk.corpus.reader.plaintext import PlaintextCorpusReader

################################################################################

# Loads the data from a set of data and assigned labels
def load_data(dir_label):
    
    data_list = []
    labels = []

    for dl in dir_label:
        
        data = []

        directory = dl[0]
        label = dl[1]
    
        corpus_dir = 'dataset/' + directory
        corpus = PlaintextCorpusReader(corpus_dir,'.*\.*')
        file_ids = corpus.fileids()
        
        for file in file_ids:
            
            d = []
            
            text = corpus.raw(file)
            e = email.message_from_string(text)
            
            if(e.is_multipart()):
                for payload in e.get_payload:
                    text = payload.get_payload
            else:
                text = e.get_payload()

            feats = [
                cf.charac_feats_extractor(text),
                wf.word_feats_extractor(text),
                syf.syntac_feats_extractor(text),
                stf.struct_feats_extractor(corpus, file, text),
                fwf.funct_word_feats_extractor(text)
            ]
    
            for f in feats:
                d.extend(list(f.values()))
                
            data.append(d)
            labels.append(label)
            
        data_list.extend(data)
        
    return [data_list, labels]

