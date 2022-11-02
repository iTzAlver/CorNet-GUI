# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
import bs4
import requests
import numpy as np
from ._generator import Generator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from basenet import BaseNetDatabase
from ..__special__ import __wikipedia_random_path__


# -----------------------------------------------------------
class WkGenerator:
    """The class HtGenerator: Creates a Real dataset from a generator and wikipedia."""
    def __init__(self, generator: Generator = None, queue: multiprocessing.Queue = None, **kwargs):
        if generator is not None:
            self.options: Generator = generator
        else:
            self.options = Generator(**kwargs)
        if not self.options:
            raise ValueError('WkGenerator: The generator is not valid. Check the input parameters.')
        self.model: SentenceTransformer = SentenceTransformer(self.options['model_url'])
        self.queue: multiprocessing.Queue = queue
        if queue is not None:
            queue.put('The connection with WkGenerator is successful.')

        self._distribution = {'train': self.options['distribution'][0], 'val': self.options['distribution'][1],
                              'test': self.options['distribution'][2]}
        x, y = self.build()
        thedb = BaseNetDatabase(x, y, self._distribution, name=self.options['name'])

        if self.options['path']:
            thedb.save(self.options['path'])

        if queue is not None:
            queue.put(f'Building WK database {self.options["number"]} / {self.options["number"]}.')
            queue.put('ENDC')

    def build(self):
        x = []
        y = []
        checkpoint_count = 0
        for i in range(0, self.options['number']):
            the_embedding: list[np.ndarray] = self._build_a_matrix()
            the_embedding_cut = self._reparse_embeddings(the_embedding, self.options['tput'])
            _y = self._create_solution([len(emb) for emb in the_embedding], self.options['tput'])
            _x = self._embedding_to_matrix(the_embedding_cut)
            x.append(_x)
            y.append(_y)
            if self.queue is not None:
                self.queue.put(f'Building WK database {i + 1} / {self.options["number"]}.')
            # Checkpoint.
            if checkpoint_count >= 7:
                checkpoint_count = 0
                BaseNetDatabase(x, y, self._distribution, name=self.options['name']).\
                    save(self.options['checkpoint_path'])
            else:
                checkpoint_count += 1
        return x, y

    def _build_a_matrix(self) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        while sum([len(embedding) for embedding in embeddings]) < self.options['tput']:
            this_embeddings: np.ndarray = np.array([])
            unparsed_text: list = self._read_block()
            parsed_text: list = self.__preprocess_string(unparsed_text, self.options['minimum_words'])

            if parsed_text:
                for parsed_block in parsed_text:
                    if parsed_block:
                        embeddings_block: np.ndarray = self.model.encode(parsed_block)
                        if len(this_embeddings) > 0:
                            this_embeddings = np.append(this_embeddings, embeddings_block, 0)
                        else:
                            this_embeddings = embeddings_block
                embeddings.append(this_embeddings)
            print(sum([len(embedding) for embedding in embeddings]))
        return embeddings

    @staticmethod
    def __preprocess_string(lines: list[str], minimum_words: int):
        _return_lines_ = []
        for line in lines:
            if 'Categorías:' not in line and '↑' not in line:
                _return_line_ = line
                _return_line_ = _return_line_.replace(' ', '').replace('  ', ' ').replace('  ', ' ')\
                    .replace('  ', ' ').replace('»', '"').replace('«', '"').replace(' ', '').replace(' ', '').\
                    replace('\u200b', '')
            else:
                _return_line_ = ''
            if len(_return_line_.split(' ')) > minimum_words:
                _return_line_ = _return_line_.replace('...', '↑. ')
                _return_sentences_ = [sentence.replace('↑', '...') for sentence in _return_line_.split('. ')]
                _return_lines_.append(_return_sentences_)
        return _return_lines_

    @staticmethod
    def _embedding_to_matrix(embeddings: np.array):
        r = np.zeros((len(embeddings), len(embeddings)))
        for ne1, embedding1 in enumerate(embeddings):
            for ne2, embedding2 in enumerate(embeddings):
                if ne1 <= ne2:
                    value = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
                    r[ne1, ne2] = value
                    r[ne2, ne1] = value
        return r

    @staticmethod
    def _read_block():
        # Obtain the text from the random URL.
        response = requests.get(__wikipedia_random_path__, headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs4.BeautifulSoup(response.text, features="html.parser")
        text_list = soup.body.get_text().split('\n')

        _text = ''
        _text_list = [(0, ''), (0, '')]
        for _text_ in text_list:
            _nwords = len(_text_.split(' '))
            _text_list.append((_nwords, _text_))
        _text_list.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in _text_list]

    @staticmethod
    def _reparse_embeddings(the_embedding: list[np.ndarray], tput) -> np.ndarray:
        this_embeddings: np.ndarray = the_embedding[0]
        for embedding in the_embedding[1:]:
            this_embeddings = np.append(this_embeddings, embedding, 0)
        return this_embeddings[:tput]

    @staticmethod
    def _create_solution(sort_offs, tput) -> np.ndarray:
        solution = np.zeros(tput)
        solution[0] = 1
        idx_acc = 0
        for sortoff in sort_offs:
            idx_acc += sortoff
            if idx_acc < tput:
                solution[idx_acc] = 1
        return solution
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
