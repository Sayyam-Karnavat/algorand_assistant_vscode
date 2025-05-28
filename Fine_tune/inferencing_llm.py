def sample(preds,temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) /temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)


def generate_text(length,temperature):
    start_index = random.randint(0,len(text)- seq_length - 1)
    generated = ''
    sentence = text[start_index : start_index + seq_length]
    generated +=sentence
    for i in range(length):
        x = np.zeros((1,seq_length,len(unique_characters)))
        for t,character in enumerate(sentence):
            x[0,t,char_to_index[character]] = 1
        prediction = model.predict(x,verbose = 0)[0]
        next_index = sample(prediction , temperature)
        next_character = index_to_char[next_index]
        
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print(generate_text(300,0.2))