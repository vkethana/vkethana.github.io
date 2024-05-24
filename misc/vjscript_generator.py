import re

def parse_vjscript(vjscript):
    # Split input into words
    words = vjscript.split()
    parsed_words = []

    # Regular expression to find consonant clusters and optional vowels
    pattern = re.compile(r"([A-Z]+)(?:\{([A-Z]+)\})?")
    
    for word in words:
        matches = pattern.findall(word)
        if not matches:
            raise ValueError("Invalid VJScript format")
        
        parsed_data = []
        for match in matches:
            consonants, vowel = match
            parsed_data.append((consonants, vowel))
        
        parsed_words.append(parsed_data)
    
    return parsed_words

def generate_html(parsed_words):
    html_output = ''
    
    for word in parsed_words:
        html_output += '<span class="word">\n'
        
        for consonants, vowel in word:
            length = len(consonants)
            for i, consonant in enumerate(consonants):
                html_output += '    <div class="consonant-vowel">\n'
                if i == length - 1 and vowel:
                    html_output += f'        <span class="vowel">{vowel}</span>\n'
                html_output += f'        <span class="consonant">{consonant}</span>\n'
                html_output += '    </div>\n'
        
        html_output += '</span>\n'
    
    return html_output

def main():
    vjscript_sentence = "TH{I}S IS U FLL P{AE}R{U}GR{AE}F IN V{EE}J{EI}SKR{I}PT. AI AEM R{AI}T{I}NG TH{I}S S{OA} TH{AE}T UTHRS K{AE}N L{UR}N T{O} R{EE}D IT. AET FRST I H{AE}NDR{OA}T M{AI} EKS{AEM}PLS. H{OWE}VR, AI R{EE}S{E}NTL{EE} D{I}Z{AI}ND A SKR{I}PT IN P{AI}TH{AW}N T{O} AWT{OA}M{EI}T TH{U} H{OA}L PR{AW}S{E}S. Y^{O} C^{AE}N F^{AI}ND IT H{EE}R: ."
    parsed_words = parse_vjscript(vjscript_sentence)
    html = generate_html(parsed_words)
    print(html)

if __name__ == "__main__":
    main()

