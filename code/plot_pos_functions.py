import matplotlib.pyplot as plt

def do_plot(x, y, word_2_pos):
  fig, _ = plt.subplots()
  fig.set_size_inches(25, 15)
  n_x, n_y = [], []
  v_x, v_y = [], []
  adv_x, adv_y = [], []
  adj_x, adj_y = [], []
  p_x, p_y = [], []
  o_x, o_y = [], []
  for ind, val in enumerate(word_2_pos.values()):
    if val == 'noun':
      n_x.append(x[ind])
      n_y.append(y[ind])
    elif val == 'verb':
      v_x.append(x[ind])
      v_y.append(y[ind])
    elif val == 'adverb':
      adv_x.append(x[ind])
      adv_y.append(y[ind])
    elif val == 'adj':
      adj_x.append(x[ind])
      adj_y.append(y[ind])
    elif val == 'pronoun':
      p_x.append(x[ind])
      p_y.append(y[ind])
    elif val == 'other':
      o_x.append(x[ind])
      o_y.append(y[ind])
    else:
      print("ERROR")
  plt.scatter(n_x, n_y, color='red', label='noun')
  plt.scatter(v_x, v_y, color='blue', label='verb')
  plt.scatter(adv_x, adv_y, color='green', label='adverb')
  plt.scatter(adj_x, adj_y, color='yellow', label='adjective')
  plt.scatter(p_x, p_y, color='purple', label='pronoun')
  plt.scatter(o_x, o_y, color='black', label='other')

  plt.legend()

  plt.show()