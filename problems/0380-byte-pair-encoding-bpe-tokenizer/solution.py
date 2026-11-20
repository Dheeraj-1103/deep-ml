def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    words = [(key.split(), freq) for key, freq in corpus.items()]

    merges = []

    for _ in range(num_merges):

        counts = {}

        # Step 1: Count adjacent pairs
        for tokens, freq in words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                counts[pair] = counts.get(pair, 0) + freq

        # Stop if no pairs exist
        if not counts:
            break

        # Step 2: Find the most frequent pair
        best = max(counts, key=counts.get)

        merges.append(best)

        # Step 3: Merge the selected pair everywhere
        new_words = []

        for tokens, freq in words:

            result = []
            i = 0

            while i < len(tokens):

                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    result.append(tokens[i] + tokens[i + 1])
                    i += 2

                else:
                    result.append(tokens[i])
                    i += 1

            new_words.append((result, freq))

        words = new_words

    return merges