import pandas as pd
import scattertext as st
import spacy
import re

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['template'] = df['template'].fillna('N/A')
    df['idea_text'] = df['idea_text'].fillna('')
    
    df['text'] = df['idea_text']  # idea text only; scattertext_meme.py uses "Template: " + template + " | Idea: " + idea_text
    
    df['category'] = df['label'].apply(lambda x: 'Class 1' if int(x) == 1 else 'Class 0')
    
    return df[['text', 'category']]

print("loading data...")
df_train = load_and_prepare_data("train_set.csv")

print("loading spacy...")
nlp = spacy.load("en_core_web_trf")

print("constructing scattertext corpus...")

corpus = st.CorpusFromPandas(
    df_train,
    category_col='category',
    text_col='text',
    nlp=nlp,
    feats_from_spacy_doc=st.FeatsFromSpacyDoc(use_lemmas=True) 
).build().compact(st.AssociationCompactor(400))

print("generating html...")
html = st.produce_scattertext_explorer(
    corpus,
    category='Class 1',
    category_name='AI',
    not_category_name='Human',
    minimum_term_frequency=10,
    pmi_threshold_coefficient=8,
    max_snippets=14,
    metadata=df_train['category']
)

def inject_css(html_str, css):
    new_html, n = re.subn(
        r'</head>',
        lambda m: css + '</head>',
        html_str,
        count=1,
        flags=re.IGNORECASE
    )
    if n > 0:
        print("✓ CSS injected before </head>")
        return new_html

    new_html, n = re.subn(
        r'<body',
        lambda m: css + '<body',
        html_str,
        count=1,
        flags=re.IGNORECASE
    )
    if n > 0:
        print("✓ CSS injected before <body>")
        return new_html

    print("⚠ CSS injected at the beginning of the file")
    return css + html_str


custom_css = """
<style>
  svg text {
    font-size: 16px;
    font-family: "Helvetica Neue", Arial, sans-serif;
  }
  svg g.tick text {
    font-size: 14px;
  }
  svg text.label,
  svg .axis-label {
    font-size: 14px;
  }
  h3 {
    font-size: 20px;
    font-weight: 700;
  }
  div.contexts a, ul li, ol li {
    font-size: 16px;
    line-height: 1.6;
  }
  .snippet, .snippet_text, .contexts {
    font-size: 16px;
    line-height: 1.6;
  }
  body {
    font-size: 16px;
  }
</style>
"""

custom_js = """
<script>
window.addEventListener('load', function() {
  var hasRun = false;
  
  function processChart() {
    if (hasRun) return;
    var texts = Array.from(document.querySelectorAll('svg text'));
    
    var headers = texts.filter(function(t) {
      var s = t.textContent.trim();
      return /^Top\\s/i.test(s) || /^Characteristic/i.test(s);
    });
    if (headers.length === 0) return;
    hasRun = true;
    
    var KEEP_TOP = 11;
    var KEEP_CHAR = KEEP_TOP * 2 + 2;   // = 22
    
    var sections = headers.map(function(header) {
      var hRect = header.getBoundingClientRect();
      var bottomY = Infinity;
      headers.forEach(function(h2) {
        if (h2 === header) return;
        var r = h2.getBoundingClientRect();
        if (Math.abs(r.left - hRect.left) < 30 && r.top > hRect.top) {
          if (r.top < bottomY) bottomY = r.top;
        }
      });
      var words = texts.filter(function(t) {
        if (t === header) return false;
        var s = t.textContent.trim();
        if (/^Top\\s/i.test(s) || /^Characteristic/i.test(s)) return false;
        var r = t.getBoundingClientRect();
        return Math.abs(r.left - hRect.left) < 30 
            && r.top > hRect.top && r.top < bottomY;
      });
      words.sort(function(a, b) {
        return a.getBoundingClientRect().top - b.getBoundingClientRect().top;
      });
      var isChar = /^Characteristic/i.test(header.textContent.trim());
      return { 
        header: header, 
        words: words, 
        x: hRect.left,
        keep: isChar ? KEEP_CHAR : KEEP_TOP
      };
    });
    
    sections.forEach(function(sec) {
      sec.words.forEach(function(t, i) {
        if (i >= sec.keep) t.style.display = 'none';
      });
    });
    
    var byCol = {};
    sections.forEach(function(s) {
      var key = Math.round(s.x / 30) * 30;
      if (!byCol[key]) byCol[key] = [];
      byCol[key].push(s);
    });
    
    Object.keys(byCol).forEach(function(key) {
      var col = byCol[key];
      col.sort(function(a, b) {
        return a.header.getBoundingClientRect().top - b.header.getBoundingClientRect().top;
      });
      
      for (var i = 1; i < col.length; i++) {
        var prev = col[i-1];
        var curr = col[i];
        var prevVisible = prev.words.slice(0, prev.keep);
        if (prevVisible.length === 0) continue;
        
        var lastBottom = prevVisible[prevVisible.length - 1].getBoundingClientRect().bottom;
        var headerTop = curr.header.getBoundingClientRect().top;
        var desiredTop = lastBottom + 25;
        var offset = desiredTop - headerTop;
        
        if (offset < -5) {
          var elements = [curr.header].concat(curr.words.slice(0, curr.keep));
          elements.forEach(function(el) {
            var t = el.getAttribute('transform') || '';
            el.setAttribute('transform', t + ' translate(0,' + offset + ')');
          });
        }
      }
    });
    
    console.log('compact processed');
  }
  
  setTimeout(processChart, 400);
  setTimeout(processChart, 1000);
  setTimeout(processChart, 2000);
});
</script>
"""

html_with_css = inject_css(html, custom_css + custom_js)

output_file = "lexical_contrast_scattertext_trf_idea_only.html"
with open(output_file, 'wb') as f:
    f.write(html_with_css.encode('utf-8'))

with open(output_file, 'r', encoding='utf-8') as f:
    content = f.read()
print("CSS injected successfully: ", "Helvetica Neue" in content)

print(f"\n Success! The visualization file has been saved as: {output_file}")
print("Please double-click the HTML file in your file manager and open it with a browser (Chrome/Edge/Safari, etc.).")