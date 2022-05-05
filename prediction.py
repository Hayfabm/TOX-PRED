from typing import Dict, List, Optional, Tuple

from biotransformers import BioTransformers
from deepchain.components import DeepChainApp
import torch.nn.functional as F
from torch import load, nn
import torch

import numpy as np

Score = Dict[str, float]
ScoreList = List[Score]

n_classes = 2


class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 100)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(32, n_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        # x = x.view(-1, x.size(0))
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout1(x)
        x = self.activation(self.fc4(x))
        return x


class App(DeepChainApp):
    """Protein toxicity Predictor App
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        # self.num_gpus = 0 if device == "cpu" else 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = BioTransformers(backend="protbert")

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "model-fold-9.pt"

        # load_model - load for pytorch model
        self.model = ProteinClassifier(n_classes).to(device)
        if self._checkpoint_filename is not None:
            state_dict = torch.load(self.get_checkpoint_path("~/TOX-PRED/"))
            self.model.load_state_dict(state_dict)

            self.model.eval()

    @staticmethod
    def score_names() -> List[str]:
        """
        Return a list of app score names
        """
        return ["ToxicityProbability"]

    def compute_scores(self, sequences_list: List[str]) -> ScoreList:
        scores_list = []
        for i in range(len(sequences_list)):
            emb = self.transformer.compute_embeddings(
                [sequences_list[i]], pool_mode=["cls"],
            )["cls"][0]

            # forward pass throught the model
            toxic_probabilities = self.model(torch.tensor(emb).float())
            print(toxic_probabilities)
            toxic_probabilities = toxic_probabilities.detach().cpu().numpy()
            scores_list.append({self.score_names()[0]: toxic_probabilities[1]})

        return scores_list


if __name__ == "__main__":

    sequences = [
        "MSQLRWWVVSQVLLLIAICSLDHSEGARVCPKIVPGLDKLRVGVDITKLDLLPLFDLGDNGFRSAVADYTCDRGQTAVVDGESFDVPDQVDSVVIESSGQQTSSVTTIKSESQISQALSISAGISVETAKAGFSSSASYAEMQEAITKYGRTVSQMSAVYTTCSANLSPNLLLGQNPLQTLSRLPSDFTADTQGYYDFIKTYGTHYFNKGKLGGMFLFTSETDMSYFQNKNSQQIEATVKATFASILSTETGGSSDESKEVIEFKESSLITSKFFGGQTNLAADGLTKWQPTIAKLPYFMSGTLSTISSLIADTTKRASMELAVKNYLLKAKVANLDRLTYIRLNSWSVGHNELRDLSAQLQNLKTKTIFSDADEKLLQSIEDQVSVPAWFSDRTTFCFRSTAVGSADQCNGQSTNTLCAEPNRYTQQYMDKTYLGDTGCRLVWKISTTESTDWFKSVKVNFRWYPTWSPCACGPVGTPFTISAPANSWTQDYLDVTNPKFGECMLQWMIEVPPTATLWAKNLEFCIDFTCGKKKQCVDANQWTEPYLDISAHEACGMSWALIAK",
        "EIIRSNFKSNLHKVYQAIEEADFFAIDGEFSGISDGPSVSALTNGFDTPEERYQKLKKHSMDFLLFQFGLCTFKYDYTDSKYITKSFNFYVFPKPFNRSSPDVKFVCQSSSIDFLASQGFDFNKVFRNGIPYLNQEEERQLREQYDEKRSQANGAGALSYVSPNTSKCPVTIPEDQKKFIDQVVEKIEDLLQSEENKNLDLEPCTGFQRKLIYQTLSWKYPKGIHVETLETEKKERYIVISKVDEEERKRREQQKHAKEQEELNDAVGFSRVIHAIANSGKLVIGHNMLLDVMHTVHQFYCPLPADLSEFKEMTTCVFPRLLDTKLMASTQPFKDIINNTSLAELEKRLKETPFNPPKVESAEGFPSYDTASEQLHEAGYDAYITGLCFISMANYLGSFLSPPKIHVSARSKLIEPFFNKLFLMRVMDIPYLNLEGPDLQPKRDHVLHVTFPKEWKTSDLYQLFSAFGNIQISWIDDTSAFVSLSQPEQVKIAVNTSKYAESYRIQTYAEYMGRKQEEKQIKRKWTEDSWKEADSKRLNPQCIPYTLQNHYYRNNSFTAPSTVGKRNLSPSQEEAGLEDGVSGEISDTELEQTDSCAEPLSEGRKKAKKLKRMKKELSPAGSISKNSPATLFEVPDTW",
        "MSEGNAAGEPSTPGGPRPLLTGARGLIGRRPAPPLTPGRLPSIRSRDLTLGGVKKKTFTPNIISRKIKEEPKEEVTVKKEKRERDRDRQREGHGRGRGRPEVIQSHSIFEQGPAEMMKKKGNWDKTVDVSDMGPSHIINIKKEKRETDEETKQILRMLEKDDFLDDPGLRNDTRNMPVQLPLAHSGWLFKEENDEPDVKPWLAGPKEEDMEVDIPAVKVKEEPRDEEEEAKMKAPPKAARKTPGLPKDVSVAELLRELSLTKEEELLFLQLPDTLPGQPPTQDIKPIKTEVQGEDGQVVLIKQEKDREAKLAENACTLADLTEGQVGKLLIRKSGRVQLLLGKVTLDVTMGTACSFLQELVSVGLGDSRTGEMTVLGHVKHKLVCSPDFESLLDHKHR",
        "GFGCPNDYPCHRHCKSIPGRAGGYCGGAHRLRCTCYR",
        "MTKADIIEGVYEKVGFSKKESAEIVELVFDTLKETLERGDKIKISGFGNFQVRQKKARVGRNPQTGKEIEISARRVLTFRPSQVLKSALNGEAPPEDHAEIDAREEAAADAAEARGEDFDEEGMEDMEG",
    ]
    app = App("cuda:0")
    scores = app.compute_scores(sequences)
    print(scores)
