from goatools.obo_parser import GODag

def filter_mf_terms(input_obo, output_obo):
    go_dag = GODag(input_obo)
    mf_terms = {go_id for go_id, term in go_dag.items() if term.namespace == 'molecular_function'}

    with open(input_obo) as fin, open(output_obo, 'w') as fout:
        keep = False
        buffer = []

        for line in fin:
            if line.strip() == "[Term]":
                keep = False
                buffer = [line]
            elif line.strip() == "":
                if keep:
                    fout.writelines(buffer)
                    fout.write("\n")
                buffer = []
            else:
                buffer.append(line)
                if line.startswith("id: "):
                    term_id = line.strip().split("id: ")[1]
                    if term_id in mf_terms:
                        keep = True

# Example usage
# filter_mf_terms("go.obo", "go_mf_only.obo")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter GO OBO file to keep only molecular function terms.")
    parser.add_argument("input_obo", help="Input GO OBO file")
    parser.add_argument("output_obo", help="Output GO OBO file with only molecular function terms")
    args = parser.parse_args()

    filter_mf_terms(args.input_obo, args.output_obo)
    print(f"Filtered {args.input_obo} (keep MF only); output: {args.output_obo}")
