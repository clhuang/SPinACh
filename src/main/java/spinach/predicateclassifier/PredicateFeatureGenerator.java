package spinach.predicateclassifier;

import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.process.WordShapeClassifier;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.*;

/**
 * Generates features for some sentence and predicate candidate, for predicate classification
 *
 * @author Calvin Huang
 */
public class PredicateFeatureGenerator {

    public static final int WORD_SHAPER = WordShapeClassifier.WORDSHAPECHRIS2;
    public static final boolean CHILD_INDICES_IN_FEATURES = false;
    public static final boolean CHILDREN_INDICES_FEATURE = false;

    Token prev2Token;
    Token prevToken;
    Token currToken;
    Token nextToken;
    Token next2Token;

    private void centerToken(TokenSentence sentence, Token token) {
        int sentenceIndex = token.sentenceIndex;

        currToken = token;
        if (sentenceIndex > 0)
            prevToken = sentence.tokenAt(sentenceIndex - 1);
        else
            prevToken = Token.emptyToken;
        if (sentenceIndex > 1)
            prev2Token = sentence.tokenAt(sentenceIndex - 2);
        else
            prev2Token = Token.emptyToken;
        if (sentenceIndex < sentence.size() - 1)
            nextToken = sentence.tokenAt(sentenceIndex + 1);
        else
            nextToken = Token.emptyToken;
        if (sentenceIndex < sentence.size() - 2)
            next2Token = sentence.tokenAt(sentenceIndex + 2);
        else
            next2Token = Token.emptyToken;
    }

    private void clearFocus() {
        prev2Token = null;
        prevToken = null;
        currToken = null;
        nextToken = null;
        next2Token = null;
    }

    /**
     * Generates a datum with features for some token and the surrrounding sentence
     *
     * @param sentence  sentence the predicate is in
     * @param predicate predicate to generate the features around
     * @return datum
     */
    public Datum<String, String> datumFrom(TokenSentenceAndPredicates sentence, Token predicate) {
        return new BasicDatum<String, String>(featuresOf(sentence, predicate));
    }

    protected Collection<String> featuresOf(TokenSentence sentence, Token predicate) {

        List<String> features = new ArrayList<String>();

        centerToken(sentence, predicate);

        List<String> predicateLemmaFeatures = lemmaFeatures();
        List<String> predicateFormFeatures = formFeatures();

        features.addAll(predicateLemmaFeatures);
        features.addAll(predicateFormFeatures);
        features.addAll(posFeatures());
        features.addAll(wordShapeFeatures());
        List<Token> children = sentence.getChildren(predicate);

        //add number of children
        features.add("numch|" + children.size());

        //add children features
        for (Token child : children) {
            int relativePosition = predicate.sentenceIndex - child.sentenceIndex;
            centerToken(sentence, child);
            for (String s : lemmaFeatures()) {
                features.add("c" + s);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("c" + relativePosition + s);
            }

            for (String s : formFeatures()) {
                features.add("c" + s);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("c" + relativePosition + s);
            }

            for (String s : posFeatures()) {
                features.add("c" + s);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("c" + relativePosition + s);
            }

            Iterator<String> parentIterator = predicateLemmaFeatures.iterator();
            Iterator<String> childIterator = lemmaFeatures().iterator();
            while (parentIterator.hasNext() && childIterator.hasNext()) {
                String childString = childIterator.next();
                String parentString = parentIterator.next();
                features.add("cp" +
                        childString + "||" +
                        parentString);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("cp" + relativePosition +
                            childString + "||" +
                            parentString);
            }

            parentIterator = predicateFormFeatures.iterator();
            childIterator = formFeatures().iterator();
            while (parentIterator.hasNext() && childIterator.hasNext()) {
                String childString = childIterator.next();
                String parentString = parentIterator.next();
                features.add("cp" +
                        childString + "||" +
                        parentString);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("cp" + relativePosition +
                            childString + "||" +
                            parentString);
            }

        }

        if (CHILDREN_INDICES_FEATURE) {
            StringBuilder s = new StringBuilder("chdif|");
            for (Token child : sentence.getChildren(predicate)) {
                s.append(predicate.sentenceIndex - child.sentenceIndex);
                s.append(" ");
            }
            features.add(s.toString());

        }

        clearFocus();

        return features;
    }

    private List<String> lemmaFeatures() {
        List<String> features = new ArrayList<String>();

        //lemma unigrams
        features.add("splmu,i-1|" + prevToken.lemma);
        features.add("splmu,i|" + currToken.lemma);
        features.add("splmu,i+1|" + nextToken.lemma);

        //lemma bigrams
        features.add("splmb,i-1,i|" +        //<i-1, i>
                prevToken.lemma + " " + currToken.lemma);
        features.add("splmb,i,i+1|" +        //<i, i+1>
                currToken.lemma + " " + nextToken.lemma);

        return features;
    }

    private List<String> formFeatures() {
        List<String> features = new ArrayList<String>();

        //form unigrams
        features.add("spfm,i-2|" + prev2Token.form);
        features.add("spfm,i-1|" + prevToken.form);
        features.add("spfm,i|" + currToken.form);
        features.add("spfm,i+1|" + nextToken.form);
        features.add("spfm,i+2|" + next2Token.form);

        return features;
    }

    private List<String> posFeatures() {
        List<String> features = new ArrayList<String>();

        //pos unigrams
        features.add("pposu,i-1|" + prevToken.pos);
        features.add("pposu,i|" + currToken.pos);
        features.add("pposu,i+1|" + nextToken.pos);

        //pos bigrams
        features.add("pposb,i-2,i-1|" +    //<i-2, i-1>
                prev2Token.pos + " " + prevToken.pos);
        features.add("pposb,i-1,i|" +        //<i-1, i>
                prevToken.pos + " " + currToken.pos);
        features.add("pposb,i,i+1|" +        //<i, i+1>
                currToken.pos + " " + nextToken.pos);
        features.add("pposb,i+1,i+2|" +    //<i, i+1>
                nextToken.pos + " " + next2Token.pos);

        return features;
    }

    private List<String> wordShapeFeatures() {
        List<String> features = new ArrayList<String>();

        String wordShape = wordShapeOf(currToken);
        String prevWordShape = wordShapeOf(prevToken);
        String prev2WordShape = wordShapeOf(prev2Token);
        String nextWordShape = wordShapeOf(nextToken);
        String next2WordShape = wordShapeOf(next2Token);

        //word shape unigrams
        features.add("wdshpu,i-1|" + prevWordShape);
        features.add("wdshpu,i|" + wordShape);
        features.add("wdshpu,i+1|" + nextWordShape);

        //word shape bigrams
        features.add("wdshpb,i-1,i|" +
                prevWordShape + " " + wordShape);
        features.add("wdshpb,i,i+1|" +
                wordShape + " " + nextWordShape);

        //word shape trigrams
        features.add("wdshpt,i-2,i-1,i|" +
                prev2WordShape + " " +
                prevWordShape + " " +
                wordShape);
        features.add("wdshpt,i,i+1,i+2|" +
                wordShape + " " +
                nextWordShape + " " +
                next2WordShape);

        return features;
    }

    private String wordShapeOf(Token t) {
        return WordShapeClassifier.wordShape(t.form, WORD_SHAPER);
    }

}
