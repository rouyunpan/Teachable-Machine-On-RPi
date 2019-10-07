"""Detection Engine used for detection tasks."""
from collections import Counter
from collections import defaultdict
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter


class EmbeddingEngine():
  """Engine used to obtain embeddings from headless mobilenets."""

  def __init__(self, model_path):
    """Creates a EmbeddingEngine with given model and labels.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    global interpreter
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()

    global input_details
    global output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
  
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    global required_image_size 
    required_image_size = (width, height)
    
    output_tensors_sizes = output_details[0]['shape'][3]
    if output_tensors_sizes != 1024:
      raise ValueError('Dectection model should have only 1024 output tensor!')
    pass

  def DetectWithImage(self, img):
    """Calculates embedding from an image.

    Args:
      img: PIL image object.

    Returns:
      Embedding vector as np.float32

    Raises:
      RuntimeError: when model's input tensor format is invalid.
    """

    with img.resize(required_image_size, Image.NEAREST) as resized_img:
      input_data = np.expand_dims(resized_img, axis=0)
      #Specifies the num of threads assigned to inference
      interpreter.set_num_threads(4) 
      interpreter.set_tensor(input_details[0]['index'], input_data)
      interpreter.invoke()     
      output_data = interpreter.get_tensor(output_details[0]['index'])
      return np.squeeze(output_data)


class KNNEmbeddingEngine(EmbeddingEngine):
  """Extends embedding engine to also provide kNearest Neighbor detection.

     This class maintains an in-memory store of embeddings and provides
     functions to find k nearest neighbors against a query emedding.
  """

  def __init__(self, model_path, kNN=3):
    """Creates a EmbeddingEngine with given model and labels.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    EmbeddingEngine.__init__(self, model_path)
    self.clear()
    self._kNN = kNN

  def clear(self):
    """Clear the store: forgets all stored embeddings."""
    self._labels = []
    self._embedding_map = defaultdict(list)
    self._embeddings = None

  def addEmbedding(self, emb, label):
    """Add an embedding vector to the store."""

    normal = emb/np.sqrt((emb**2).sum()) # Normalize the vector

    self._embedding_map[label].append(normal) # Add to store, under "label"

    # Expand labelled blocks of embeddings for when we have less than kNN
    # examples. Otherwise blocks that have more examples unfairly win.
    emb_blocks = []
    self._labels = [] # We'll be reconstructing the list of labels
    for label, embeds in self._embedding_map.items():
      emb_block = np.stack(embeds)
      if emb_block.shape[0] < self._kNN:
          emb_block = np.pad(emb_block,
                             [(0,self._kNN - emb_block.shape[0]), (0,0)],
                             mode="reflect")
      emb_blocks.append(emb_block)
      self._labels.extend([label]*emb_block.shape[0])

    self._embeddings = np.concatenate(emb_blocks, axis=0)

  def kNNEmbedding(self, query_emb):
    """Returns the self._kNN nearest neighbors to a query embedding."""

    # If we have nothing stored, the answer is None
    if self._embeddings is None: return None

    # Normalize query embedding
    query_emb = query_emb/np.sqrt((query_emb**2).sum())

    # We want a cosine distance ifrom query to each stored embedding. A matrix
    # multiplication can do this in one step, resulting in a vector of
    # distances.
    dists = np.matmul(self._embeddings, query_emb)

    # If we have less than self._kNN distances we can only return that many.
    kNN = min(len(dists), self._kNN)

    # Get the N largest cosine similarities (larger means closer).
    n_argmax = np.argpartition(dists, -kNN)[-kNN:]

    # Get the corresponding labels associated with each distance.
    labels = [self._labels[i] for i in n_argmax]

    # Return the most common label over all self._kNN nearest neighbors.
    most_common_label = Counter(labels).most_common(1)[0][0]
    return most_common_label

  def exampleCount(self):
    """Just returns the size of the embedding store."""
    return sum(len(v) for v in self._embedding_map.values())


