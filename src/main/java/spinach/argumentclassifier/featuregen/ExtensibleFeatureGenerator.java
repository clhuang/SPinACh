package spinach.argumentclassifier.featuregen;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.*;

/**
 * This feature generator, in addition to the argument feature generator,
 * enables addition of user-defined custom features
 *
 * @author Calvin Huang
 */
public class ExtensibleFeatureGenerator extends ArgumentFeatureGenerator {

    private static final long serialVersionUID = -6330635591411786608L;

    private static final String ARGUMENT = "argument";
    private static final String PREDICATE = "predicate";

    private Set<IndividualFeatureGenerator> enabledFeatures;
    private final Set<IndividualFeatureGenerator> featureGeneratorSet =
            new HashSet<IndividualFeatureGenerator>();

    /**
     * Generates a new ExtensibleFeatureGenerator, and adds
     * default features to the set of possible features.
     */
    public ExtensibleFeatureGenerator() {
        addDefaultFeatures();
    }

    @Override
    protected Collection<String> featuresOf(SemanticFrameSet sentence,
                                            Token argument, Token predicate) {

        Collection<String> features = super.featuresOf(sentence, argument, predicate);
        Map<String, Token> featureTokens = new HashMap<String, Token>();
        /* featureToken tokens:
        argument
        predicate
        ppHead
         */

        featureTokens.put(ARGUMENT, argument);
        featureTokens.put(PREDICATE, predicate);


        for (IndividualFeatureGenerator featureGenerator : enabledFeatures) {
            Collection<String> newFeatures = featureGenerator.
                    featuresOf(sentence, featureTokens);
            if (newFeatures != null)
                features.addAll(newFeatures);
        }

        return features;

    }

    /**
     * Disables all extra features
     */
    public void clearFeatures() {
        enabledFeatures.clear();
    }

    /**
     * Adds some feature generator to the list of feature generators
     *
     * @param featureGenerator feature generator to be added
     * @return index of feature generator in list
     */
    public int addFeature(IndividualFeatureGenerator featureGenerator) {
        featureGeneratorSet.add(featureGenerator);
        return featureGeneratorSet.size() - 1;
    }

    private void addDefaultFeatures() {
        addFeature(new IndividualFeatureGenerator("existSemDeprel") {

            @Override
            public Collection<String> featuresOf(SemanticFrameSet frameSet,
                                                 Map<String, Token> featureTokens) {
                //TODO
                return null;
            }
        });

        addFeature(new IndividualFeatureGenerator("previousArgClass") {

            @Override
            public Collection<String> featuresOf(SemanticFrameSet frameSet, Map<String, Token> featureTokens) {

                int mostRecentArgumentIndex = -1;
                String mostRecentLabel = null;

                Token predicate = featureTokens.get(PREDICATE);
                for (Map.Entry<Token, String> pair : frameSet.argumentsOf(predicate).entrySet()) {
                    int currIndex = pair.getKey().sentenceIndex;
                    if (currIndex > mostRecentArgumentIndex) {
                        mostRecentArgumentIndex = currIndex;
                        mostRecentLabel = pair.getValue();
                    }
                }

                if (mostRecentLabel == null)
                    return Collections.singletonList("previousArgClass:" + ArgumentClassifier.NIL_LABEL);
                else
                    return Collections.singletonList("previousArgClass:" + mostRecentLabel);
            }

        });

        addFeature(new IndividualFeatureGenerator("linePath") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Map<String, Token> featureTokens) {
                StringBuilder linePathF = new StringBuilder("linePathF|");
                StringBuilder linePathL = new StringBuilder("linePathL|");
                StringBuilder linePathD = new StringBuilder("linePathD|");

                Token argument = featureTokens.get(ARGUMENT);
                Token predicate = featureTokens.get(PREDICATE);

                Token start;
                Token end;

                if (predicate.comesBefore(argument)) {
                    start = predicate;
                    end = argument;
                } else {
                    start = argument;
                    end = predicate;
                }

                for (Token t = start; !end.comesBefore(t);
                     t = frameSet.tokenAt(t.sentenceIndex + 1)) {
                    linePathF.append(t.form).append(" ");
                    linePathL.append(t.lemma).append(" ");
                    linePathD.append(t.syntacticHeadRelation).append(" ");
                }

                return new ImmutableList.Builder<String>().add(linePathF.toString(),
                        linePathL.toString(),
                        linePathD.toString()).build();
            }
        });

        addFeature(new IndividualFeatureGenerator("dpTreeRelation") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Map<String, Token> featureTokens) {
                //TODO
                return null;
            }
        });

        addFeature(new IndividualFeatureGenerator("hi/lo support") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Map<String, Token> featureTokens) {
                Token hiVerb = Token.emptyToken;
                Token loVerb = Token.emptyToken;
                Token hiNoun = Token.emptyToken;
                Token loNoun = Token.emptyToken;

                Token argument = featureTokens.get(ARGUMENT);

                for (Token token : frameSet.ancestorPath(argument, frameSet.getRoot())) {
                    if (token.isNoun()) {
                        if (hiNoun == Token.emptyToken || hiNoun.comesBefore(token))
                            hiNoun = token;
                        if (loNoun == Token.emptyToken || token.comesBefore(loNoun))
                            loNoun = token;
                    } else if (token.isVerb()) {
                        if (hiVerb == Token.emptyToken || hiVerb.comesBefore(token))
                            hiVerb = token;
                        if (loVerb == Token.emptyToken || token.comesBefore(loVerb))
                            loVerb = token;
                    }
                }

                return new ImmutableList.Builder<String>().add(
                        "argHiNF|" + hiNoun.form,
                        "argHiNL|" + hiNoun.lemma,
                        "argHiNP|" + hiNoun.pos,
                        "argLoNF|" + loNoun.form,
                        "argLoNL|" + loNoun.lemma,
                        "argLoNP|" + loNoun.pos,
                        "argHiVF|" + hiVerb.form,
                        "argHiVL|" + hiVerb.lemma,
                        "argHiVP|" + hiVerb.pos,
                        "argLoVF|" + loVerb.form,
                        "argLoVL|" + loVerb.lemma,
                        "argLoVP|" + loVerb.pos
                ).build();
            }
        });

        addFeature(new IndividualFeatureGenerator("isArgLeaf") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Map<String, Token> featureTokens) {
                return Collections.singletonList(frameSet.getChildren(featureTokens.get(ARGUMENT)).isEmpty() ?
                        "argLeaf" :
                        "argNotLeaf");
            }
        });
    }

    /**
     * Returns a copy of the set of feature generators.
     *
     * @return set of feature generators
     */
    public Set<IndividualFeatureGenerator> featureGeneratorSet() {
        return Collections.unmodifiableSet(featureGeneratorSet);
    }

    /**
     * Changes the set of enabled feature generators.
     *
     * @param featureGenerators new set of enabled feature generators.
     */
    public void setEnabledFeatureGenerators(Set<IndividualFeatureGenerator> featureGenerators) {
        enabledFeatures = new HashSet<IndividualFeatureGenerator>(featureGenerators);
    }

    /**
     * Returns a copy of the set of enabled feature generators.
     *
     * @return set of enabled feature generators
     */
    public Set<IndividualFeatureGenerator> enabledFeatures() {
        return Collections.unmodifiableSet(enabledFeatures);
    }

    /**
     * Returns a set of disabled feature generators.
     *
     * @return difference between featureGeneratorSet() and enabledFeatures()
     */
    public Set<IndividualFeatureGenerator> disabledFeatures() {
        return Sets.difference(featureGeneratorSet, enabledFeatures);
    }

}

