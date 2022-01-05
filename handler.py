import numpy as np
import torch
import torch.nn as nn

from model import AttentionModel


class ModelHandler(object):

    def __init__(self):
        self.initialized = False
        self.device = 'cpu'
        self.hidden_size = 512
        self.emb_size = 256
        self.text = torch.load('./source_vocab.pt')
        self.label = torch.load('./label_obj.pt')
        self._model = None
        self.input_size = len(self.text.vocab)
        self.output_size = len(self.label.vocab)

    def initialize(self, context):
        self.manifest = context.manifest
        self._model = AttentionModel(self.output_size, self.hidden_size, self.input_size, self.emb_size)
        self._model.load_state_dict(torch.load(self.manifest["model"]["serializedFile"],
                                               map_location=torch.device(self.device)))
        self._model = self._model.to(self.device)
        self._model.eval()
        self.initialized = True

    def postprocess(self, predicted):
        result = [{
            "sentiment":
                {
                    "label": predicted
                }
        }]
        return result

    def handle(self, data, context):
        test_sen = self.text.preprocess(str(data[0]['body'].decode("utf-8")))
        test_sen = [[self.text.vocab.stoi[x] for x in test_sen]]
        test_sen = np.asarray(test_sen)
        with torch.no_grad():
            output = self._model(torch.LongTensor(test_sen).permute(1, 0).to(self.device))
        _, preds = torch.max(output, 1)
        s = nn.Softmax(dim=1)
        label = 'positive' if preds.cpu().numpy()[0] == 1 else 'negative'
        return self.postprocess(label)
