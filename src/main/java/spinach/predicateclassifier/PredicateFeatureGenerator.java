package spinach.predicateclassifier;

import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.process.WordShapeClassifier;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.io.Serializable;
import java.util.*;

/**
 * Generates features for some sentence and predicate candidate, for predicate classification
 *
 * @author Calvin Huang
 */
public class PredicateFeatureGenerator implements Serializable {

    private static final long serialVersionUID = -1845631435481366524L;

    private static final int WORD_SHAPER = WordShapeClassifier.WORDSHAPECHRIS2;
    private static final boolean CHILD_INDICES_IN_FEATURES = false;
    private static final boolean CHILDREN_INDICES_FEATURE = false;

    private static final int FEATURE_COUNT_THRESHOLD = 3;

    private transient Token prev2Token;
    private transient Token prevToken;
    private transient Token currToken;
    private transient Token nextToken;
    private transient Token next2Token;

    private Set<String> allowedNonStructuralFeatures;

    private static final String STRUCTURAL_FEATURE_PREFIX = "QQ";

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
     * Generates a datum with features for some token and the surrounding sentence
     *
     * @param sentence  sentence the predicate is in
     * @param predicate predicate to generate the features around
     * @return datum
     */
    public Datum<String, String> datumFrom(TokenSentenceAndPredicates sentence, Token predicate) {
        return new BasicDatum<String, String>(reducedFeaturesOf(sentence, predicate));
    }

    /**
     * Generates features for some sentence and some predicate
     *
     * @param sentence  sentence containing predicate
     * @param predicate predicate candidates to generate feature for
     * @return datum (without label) for this sentence, predicate
     */
    Collection<String> reducedFeaturesOf(TokenSentenceAndPredicates sentence, Token predicate) {
        if (allowedNonStructuralFeatures == null)
            return featuresOf(sentence, predicate);

        Collection<String> features = new ArrayList<String>();
        for (String s : featuresOf(sentence, predicate))
            if (isStructuralFeature(s) || allowedNonStructuralFeatures.contains(s))
                features.add(s);

        return features;
    }

    private boolean isStructuralFeature(String s) {
        return s.startsWith(STRUCTURAL_FEATURE_PREFIX);
    }

    protected Collection<String> featuresOf(TokenSentenceAndPredicates sentence, Token predicate) {

        Collection<String> features = new ArrayList<String>();

        centerToken(sentence, predicate);

        List<String> predicateLemmaFeatures = lemmaFeatures();
        List<String> predicatePOSFeatures = posFeatures();

        features.addAll(predicateLemmaFeatures);
        features.addAll(formFeatures());
        features.addAll(predicatePOSFeatures);
        features.addAll(wordShapeFeatures());

        List<Token> children = sentence.getChildren(predicate);

        //add number of children
        features.add("numch|" + children.size());

        //add children features
        for (Token child : children) {
            int relativePosition = predicate.sentenceIndex - child.sentenceIndex;

            centerToken(sentence, child);

            List<String> childLemmaFeatures = lemmaFeatures();
            List<String> childPOSFeatures = posFeatures();

            //child lemma features
            for (String s : childLemmaFeatures) {
                features.add("c" + s);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("c" + relativePosition + s);
            }

            //child pos feature
            for (String s : childPOSFeatures) {
                features.add("c" + s);
                if (CHILD_INDICES_IN_FEATURES)
                    features.add("c" + relativePosition + s);
            }

            //child syntactic relation
            features.add("crel|" + child.syntacticHeadRelation);
            if (CHILD_INDICES_IN_FEATURES)
                features.add("crel|" + relativePosition + child.syntacticHeadRelation);

            //concatenation of child, parent forms/pos features
            Iterator<String> parentIterator = predicateLemmaFeatures.iterator();
            Iterator<String> childIterator = childLemmaFeatures.iterator();
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

            parentIterator = predicatePOSFeatures.iterator();
            childIterator = childPOSFeatures.iterator();
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
            StringBuilder s = new StringBuilder("cdif|");
            for (Token child : sentence.getChildren(predicate)) {
                s.append(predicate.sentenceIndex - child.sentenceIndex);
                s.append(' ');
            }
            features.add(s.toString());

        }

        clearFocus();

        return features;
    }

    private List<String> lemmaFeatures() {
        List<String> features = new ArrayList<String>();

        //lemma unigrams
        features.add("l-1|" + prevToken.lemma);
        features.add("l0|" + currToken.lemma);
        features.add("l1|" + nextToken.lemma);

        //lemma bigrams
        features.add("l-10|" +        //<i-1, i>
                prevToken.lemma + ' ' + currToken.lemma);
        features.add("l01|" +        //<i, i+1>
                currToken.lemma + ' ' + nextToken.lemma);

        return features;
    }

    private List<String> formFeatures() {
        List<String> features = new ArrayList<String>();

        //form unigrams
        features.add("f-2|" + prev2Token.form);
        features.add("f-1|" + prevToken.form);
        features.add("f0|" + currToken.form);
        features.add("f1|" + nextToken.form);
        features.add("f2|" + next2Token.form);

        return features;
    }

    private List<String> posFeatures() {
        List<String> features = new ArrayList<String>();

        //pos unigrams
        features.add("p-1|" + prevToken.pos);
        features.add("p0|" + currToken.pos);
        features.add("p1|" + nextToken.pos);

        //pos bigrams
        features.add("p-21|" +    //<i-2, i-1>
                prev2Token.pos + ' ' + prevToken.pos);
        features.add("p-10|" +        //<i-1, i>
                prevToken.pos + ' ' + currToken.pos);
        features.add("p01|" +        //<i, i+1>
                currToken.pos + ' ' + nextToken.pos);
        features.add("p12|" +    //<i, i+1>
                nextToken.pos + ' ' + next2Token.pos);

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
        features.add("w-1|" + prevWordShape);
        features.add("w0|" + wordShape);
        features.add("w1|" + nextWordShape);

        //word shape bigrams
        features.add("w-10|" +
                prevWordShape + ' ' + wordShape);
        features.add("w01|" +
                wordShape + ' ' + nextWordShape);

        //word shape trigrams
        features.add("w-210|" +
                prev2WordShape + ' ' +
                prevWordShape + ' ' +
                wordShape);
        features.add("w012|" +
                wordShape + ' ' +
                nextWordShape + ' ' +
                next2WordShape);

        return features;
    }

    private String wordShapeOf(Token t) {
        return WordShapeClassifier.wordShape(t.form, WORD_SHAPER);
    }

    /**
     * To save memory and time, the feature generator should ignore features that
     * don't appear often in training. (This ignores so-called "structural features",
     * which are features created based on results of previously done predicate classification.)
     * <p/>
     * This generates an internal list of features that are not ignored, based on the set of training frames--
     * the set of training frames should be the same set used to train the predicate classifier.
     *
     * @param trainingSet training set used to train
     */
    public void reduceFeatureSet(Collection<SemanticFrameSet> trainingSet) {
        Counter<String> featureCounter = new ClassicCounter<String>();

        for (SemanticFrameSet frameSet : trainingSet)
            for (Token t : frameSet)
                for (String s : featuresOf(frameSet, t))
                    featureCounter.incrementCount(s);

        System.out.println("predclass initial feature set size: " + featureCounter.size());

        allowedNonStructuralFeatures = new HashSet<String>();
        for (String s : featureCounter.keySet())
            if (featureCounter.getCount(s) >= FEATURE_COUNT_THRESHOLD)
                allowedNonStructuralFeatures.add(s);

        System.out.println("predclass final feature set size: " + allowedNonStructuralFeatures.size());
    }

    /**
     * Returns a view of the set of allowed non-structural features.
     *
     * @return set of allowed features
     */
    public Set<String> getAllowedNonStructuralFeatures() {
        return Collections.unmodifiableSet(allowedNonStructuralFeatures);
    }
}
