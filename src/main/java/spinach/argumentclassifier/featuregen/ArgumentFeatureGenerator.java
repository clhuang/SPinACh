package spinach.argumentclassifier.featuregen;

import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;

import java.io.Serializable;
import java.util.*;

/**
 * Generates features for some sentence, argument, and predicate for argument classification
 *
 * @author Calvin Huang
 */
public class ArgumentFeatureGenerator implements Serializable {

    private static final long serialVersionUID = -976444842348771794L;

    /**
     * Features are grouped into structural and non-structural feature groups--
     * structural features are features that are dependent on previously classified
     * arguments (in the same sentence), while non-structural features are not.
     * <p/>
     * All structural features should be prefixed with this structural feature prefix.
     */
    public static final String STRUCTURAL_FEATURE_PREFIX = "QQ";

    /**
     * Any non-structural features appearing less than a certain number of times
     * will be ignored.
     */
    private static final int FEATURE_COUNT_THRESHOLD = 3;

    private Set<String> allowedNonStructuralFeatures = new HashSet<String>();
    private boolean allowStructuralFeatures;

    /**
     * Generates a datum with features for some sentence, some argument, and some predicate.
     * To save time and memory, non-structural features that are not contained in an internal
     * set of allowed features are ignored.
     * <p/>
     * Before calling this, you should call reduceFeatureSet() to generate
     * the list of allowed non-structural features. If reduceFeatureSet() hasn't
     * been called previously, this will not eliminate any features..
     *
     * @param frameSet  SemanticFrameSet used to generate features
     * @param argument  possible argument to generate features for
     * @param predicate predicate that the argument is the argument of
     * @return datum (without label) for this sentence, predicate and argument
     */
    public Datum<String, String> datumFrom(SemanticFrameSet frameSet,
                                           Token argument, Token predicate) {
        return new BasicDatum<String, String>(reducedFeaturesOf(frameSet, argument, predicate));
    }

    /**
     * Generates a list of features for a sentence, argument, and predicate.
     * To save time and memory, it ignores non-structural features that are not
     * contained in an internal set of allowed features.
     * <p/>
     * Before calling this, you should call reduceFeatureSet() to generate
     * the list of allowed non-structural features. If reduceFeatureSet() hasn't
     * been called previously, this will just return featuresOf().
     *
     * @param sentence  sentence of predicate and argument
     * @param argument  argument candidate to generate features for
     * @param predicate predicate of argument
     * @return collection of features
     */
    final Collection<String> reducedFeaturesOf(SemanticFrameSet sentence,
                                               Token argument, Token predicate) {

        if (allowedNonStructuralFeatures == null)
            return featuresOf(sentence, argument, predicate);

        Predicate<String> filter = new Predicate<String>() {
            @Override
            public boolean apply(String input) {
                return allowedNonStructuralFeatures.contains(input) ||
                        (isStructuralFeature(input) && allowStructuralFeatures);
            }
        };

        return Collections2.filter(featuresOf(sentence, argument, predicate), filter);
    }

    private boolean isStructuralFeature(String s) {
        return s.startsWith(STRUCTURAL_FEATURE_PREFIX);
    }

    /**
     * Generates a list of features for a sentence, argument, and predicate.
     *
     * @param sentence  sentence of predicate and argument
     * @param argument  argument candidate to generate features for
     * @param predicate predicate of argument
     * @return collection of features
     */
    protected Collection<String> featuresOf(SemanticFrameSet sentence,
                                            Token argument, Token predicate) {

        Collection<String> features = new HashSet<String>();

        /*
		 * Feature 1: argument (and modifier), predicate split lemma, form; pos
		 */
        features.add("aL|" + argument.lemma);
        features.add("aF|" + argument.form);
        features.add("aP|" + argument.pos);

        features.add("pL|" + predicate.lemma);
        features.add("pF|" + predicate.form);
        features.add("pP|" + predicate.pos);

        Token pmod = getPMOD(sentence, argument);
        if (pmod != null) {
            features.add("pmodL|" + pmod.lemma);
            features.add("pmodF|" + pmod.form);
            features.add("pmodP|" + pmod.pos);
        }

        Token ppHead;
        if (argument.syntacticHeadRelation.equals("PMOD"))
            ppHead = sentence.getLeftSiblings(sentence.getParent(argument)).getFirst();
        else
            ppHead = sentence.getParent(argument);

        if (ppHead == null)
            ppHead = Token.emptyToken;

        features.add("ppHeadL|" + ppHead.lemma);
        features.add("ppHeadF|" + ppHead.form);
        features.add("aL+ppHeadF|" + argument.lemma + "|" + ppHead.form);


        /*
        * Feature 2: pos/deprel for predicate children, children of predicate ancestor across VC/IM dependencies
        */
        StringBuilder relationFeature = new StringBuilder("predCRel|");
        StringBuilder posFeature = new StringBuilder("predCPos|");
        for (Token child : sentence.getChildren(predicate)) {
            relationFeature.append(child.syntacticHeadRelation).append(" ");
            posFeature.append(child.pos).append(" ");
        }
        features.add(relationFeature.toString());
        features.add(posFeature.toString());

        relationFeature = new StringBuilder("vcimRel|");
        posFeature = new StringBuilder("vcimPos|");
        Token vcimAncestor = predicate;
        while (vcimAncestor.syntacticHeadRelation.equals("VC") || vcimAncestor.syntacticHeadRelation.equals("IM"))
            vcimAncestor = sentence.getParent(vcimAncestor);

        for (Token ancestorChild : sentence.getChildren(vcimAncestor)) {
            relationFeature.append(" ").append(ancestorChild.syntacticHeadRelation);
            posFeature.append(" ").append(ancestorChild.pos);
            if (ancestorChild.equals(argument)) {
                relationFeature.append("a");
                posFeature.append("a");
            } else if (ancestorChild.equals(predicate)) {
                relationFeature.append("p");
                posFeature.append("p");
            }
        }
        features.add(relationFeature.toString());
        features.add(posFeature.toString());


        /*
        * Feature 3: dependency path
        */
        StringBuilder path = new StringBuilder();

        /*
        * Ancestor splits the path into two halves:
        * from the argument to the ancestor, the dependencies go upwards;
        * from the predicate to the ancestor, the dependencies go downwards
        */
        Token ancestor = sentence.getCommonAncestor(argument, predicate);

        Deque<Token> argPath = sentence.ancestorPath(argument, ancestor);
        Deque<Token> predPath = sentence.ancestorPath(predicate, ancestor);

        argPath.removeLast(); //ancestor is last thing in both pathA and pathB, don't need it
        predPath.removeLast();

        while (!argPath.isEmpty())    //argPath is (upward) path from argument to ancestor
            path.append(argPath.removeFirst().syntacticHeadRelation).append("^ ");
        while (!predPath.isEmpty())    //predPath is (downwards) path from predicate to ancestor
            path.append(predPath.removeLast().syntacticHeadRelation).append("v ");

        features.add("path|" + path.toString());
        features.add("pathP|" + argument.pos + " " + path.toString() + predicate.pos);    //with pos tags
        features.add("pathL|" + argument.lemma + " " + path.toString() + predicate.lemma);    //with lemma tags

        //p.voice + a:p|direction
        String predicateVoice = sentence.voiceOf(predicate);
        features.add("pV+apDir|" + predicateVoice + "|" +
                (predicate.sentenceIndex > argument.sentenceIndex ? "pFirst" : "aFirst"));

        /*
		 * Feature 4: length of dependency path
		 */
        features.add("pathLength|" + (argPath.size() + predPath.size()));

        /*
           * Feature 5: difference in positions, and binary tokens
           */
        int distance = Math.abs(predicate.sentenceIndex - argument.sentenceIndex);
        features.add("dist|" + distance);
        features.add(distance == 1 ? "dist=1|t" : "dist=1|f");
        features.add(distance == 2 ? "dist=2|t" : "dist=2|f");
        features.add(distance > 2 ? "dist>2|t" : "dist>2|f");

        /*
           * Feature 6: predicate before or after argument
           */
        features.add("predRelPos|" + ((predicate.sentenceIndex < argument.sentenceIndex) ? "before" : "after"));

        /*
         * other features
         */

        //is current predicate
        if (argument.equals(predicate))
            features.add("isCurPred|" + predicate.lemma);
        else
            features.add("isCurPred|nope");

        //predicate voice
        features.add("pVoice|" + predicateVoice);

        return features;
    }

    private static Token getPMOD(TokenSentence sentence, Token argument) {
        for (Token t : sentence.getChildren(argument))
            if (t.syntacticHeadRelation.startsWith("PMOD"))
                return t;
        return null;
    }

    /**
     * To save memory and time, the feature generator should ignore features that
     * don't appear often in training. (This ignores so-called "structural features",
     * which are features created based on results of previously done argument classification.)
     * <p/>
     * This generates an internal list of features that are not ignored, based on the set of training frames--
     * the set of training frames should be the same set used to train the argument classifier.
     *
     * @param trainingSet training set used to train
     */
    public void reduceFeatureSet(Collection<SemanticFrameSet> trainingSet) {
        Counter<String> featureCounter = new ClassicCounter<String>();

        for (SemanticFrameSet frameSet : trainingSet)
            for (Token predicate : frameSet.getPredicateList())
                for (Token argument : ArgumentClassifier.argumentCandidates(frameSet, predicate))
                    for (String s : featuresOf(frameSet, argument, predicate))
                        if (!isStructuralFeature(s))
                            featureCounter.incrementCount(s);

        allowedNonStructuralFeatures = new HashSet<String>();
        for (String s : featureCounter.keySet())
            if (featureCounter.getCount(s) >= FEATURE_COUNT_THRESHOLD)
                allowedNonStructuralFeatures.add(s);
    }

    /**
     * Returns a view of the set of allowed non-structural features.
     *
     * @return set of allowed features
     */
    public Set<String> getAllowedNonStructuralFeatures() {
        return Collections.unmodifiableSet(allowedNonStructuralFeatures);
    }

    /**
     * Whether or not to allow structural features.
     *
     * @param allow true to allow, false otherwise
     */
    public void setAllowStructuralFeatures(boolean allow) {
        allowStructuralFeatures = allow;
    }
}
