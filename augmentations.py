import random
import flair


class TokenList:
    is_written = False
    token_list = []
    token_len = 0
    char_list = []
    char_len = 0


def augment(sentences, method, augment_p):
    # store token/char lists with global variables
    if not TokenList.is_written:
        TokenList.is_written = True
        token_list = []
        for sentence in sentences:
            for token in sentence:
                token_list.append(token.text)
        TokenList.token_list = list(set(token_list))
        TokenList.token_len = len(TokenList.token_list)

        char_list = []
        for token in token_list:
            for char in token:
                char_list.append(char)

        TokenList.char_list = list(set(char_list))
        TokenList.char_len = len(TokenList.char_list)

    method = method.split('+')
    result = []
    for i in range(len(sentences)):
        sentence = flair.data.Sentence()
        for token in sentences[i]:
            text = token.text

            if random.random() < augment_p:
                method_index = random.randint(0, len(method) - 1)
                method_act = method[method_index]

                if method_act == 'word_replace':
                    text = TokenList.token_list[random.randint(0, TokenList.token_len - 1)]
                elif method_act == 'char_replace':
                    text_list = list(text)
                    text_list[random.randint(0, len(text_list) - 1)] = TokenList.char_list[
                        random.randint(0, TokenList.char_len - 1)]
                    text = ''.join(text_list)

            nt = flair.data.Token(text)
            sentence.add_token(nt)

        result.append(sentence)

    return result
