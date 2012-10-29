package spinach.argumentclassifier.featuregen;

import com.google.common.collect.ImmutableSet;
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

    private Set<IndividualFeatureGenerator> enabledFeatures = new HashSet<IndividualFeatureGenerator>();
    private final Set<IndividualFeatureGenerator> featureGeneratorSet =
            new HashSet<IndividualFeatureGenerator>();

    /**
     * Generates a new ExtensibleFeatureGenerator, and adds
     * default features to the set of possible features.
     */
    public ExtensibleFeatureGenerator() {
        addDefaultFeatures();

        enabledFeatures.addAll(featureGeneratorSet());
    }

    @Override
    protected Collection<String> featuresOf(SemanticFrameSet sentence,
                                            Token argument, Token predicate) {

        Collection<String> features = super.featuresOf(sentence, argument, predicate);

        for (IndividualFeatureGenerator featureGenerator : enabledFeatures) {
            Collection<String> newFeatures = featureGenerator.
                    featuresOf(sentence, predicate, argument);
            if (newFeatures != null)
                features.addAll(newFeatures);
        }

        return features;
    }

    /**
     * Disables all extra features.
     */
    public void clearFeatures() {
        enabledFeatures.clear();
    }

    /**
     * Adds some feature generator to the list of feature generators.
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
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                Set<String> features = new HashSet<String>();

                for (String argLabel : frameSet.argumentsOf(predicate).values())
                    features.add("prevSR|" + argLabel);

                return features;
            }
        });

        addFeature(new IndividualFeatureGenerator("existCross") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                for (Token otherPredicate : frameSet.getPredicateList()) {
                    if (otherPredicate.equals(predicate) || otherPredicate.equals(argument))
                        continue;
                    for (Token otherArg : frameSet.argumentsOf(predicate).keySet()) {
                        if (otherArg.equals(predicate) || otherArg.equals(argument))
                            continue;
                        if (existCross(predicate, argument, otherPredicate, otherArg))
                            return Collections.singleton("existX:yes");
                    }
                }

                return Collections.singleton("existX:no");
            }

            private boolean existCross(Token predicate1, Token argument1,
                                       Token predicate2, Token argument2) {
                return (liesBetween(predicate1, predicate2, argument2) !=
                        liesBetween(argument1, predicate2, argument2));
            }

            private boolean liesBetween(Token a, Token b, Token c) {
                return ((a.comesBefore(b) && c.comesBefore(a)) ||
                        (a.comesBefore(c) && b.comesBefore(a)));
            }
        });

        addFeature(new IndividualFeatureGenerator("previousArgClass") {

            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {

                int mostRecentArgumentIndex = -1;
                String mostRecentLabel = null;

                for (Map.Entry<Token, String> pair : frameSet.argumentsOf(predicate).entrySet()) {
                    int currIndex = pair.getKey().sentenceIndex;
                    if (currIndex > mostRecentArgumentIndex) {
                        mostRecentArgumentIndex = currIndex;
                        mostRecentLabel = pair.getValue();
                    }
                }

                if (mostRecentLabel == null)
                    return Collections.singleton("previousArgClass:" + ArgumentClassifier.NIL_LABEL);
                else
                    return Collections.singleton("previousArgClass:" + mostRecentLabel);
            }

        });

        addFeature(new IndividualFeatureGenerator("linePath") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                StringBuilder linePathF = new StringBuilder("linePathF|");
                StringBuilder linePathL = new StringBuilder("linePathL|");
                StringBuilder linePathD = new StringBuilder("linePathD|");

                Token start;
                Token end;

                if (predicate.comesBefore(argument)) {
                    start = predicate;
                    end = argument;
                } else {
                    start = argument;
                    end = predicate;
                }

                for (Token t = start; t.comesBefore(end);
                     t = frameSet.tokenAt(t.sentenceIndex + 1)) {
                    linePathF.append(t.form).append(" ");
                    linePathL.append(t.lemma).append(" ");
                    linePathD.append(t.syntacticHeadRelation).append(" ");
                }

                return ImmutableSet.of(linePathF.toString(),
                        linePathL.toString(),
                        linePathD.toString());
            }
        });

        addFeature(new IndividualFeatureGenerator("dpTreeRelation") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                if (argument.equals(frameSet.getParent(predicate)))
                    return Collections.singleton("treeRel|PChild");

                if (predicate.equals(frameSet.getParent(argument)))
                    return Collections.singleton("treeRel|AChild");

                for (Token t = frameSet.getParent(argument); t != null; t = frameSet.getParent(t))
                    if (predicate.equals(t))
                        return Collections.singleton("treeRel|ADesc");

                for (Token t = frameSet.getParent(predicate); t != null; t = frameSet.getParent(t))
                    if (argument.equals(t))
                        return Collections.singleton("treeRel|PDesc");

                if (frameSet.getSiblings(predicate).contains(argument))
                    return Collections.singleton("treeRel|siblings");

                return Collections.singleton("treeRel|none");
            }
        });

        addFeature(new IndividualFeatureGenerator("hi/lo support") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                Token hiVerb = Token.emptyToken;
                Token loVerb = Token.emptyToken;
                Token hiNoun = Token.emptyToken;
                Token loNoun = Token.emptyToken;

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

                return ImmutableSet.of(
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
                );
            }
        });

        addFeature(new IndividualFeatureGenerator("isArgLeaf") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                return Collections.singleton(frameSet.getChildren(argument).isEmpty() ?
                        "argLeaf" :
                        "argNotLeaf");
            }
        });

        addFeature(new IndividualFeatureGenerator("dpPathLemma") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                StringBuilder s = new StringBuilder("PAPathLs|");

                for (Token t : frameSet.syntacticPath(predicate, argument))
                    s.append(t.lemma).append(" ");

                return Collections.singleton(s.toString());
            }
        });

        addFeature(new IndividualFeatureGenerator("T9Combo") {
            @Override
            Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
                Token argHead = frameSet.getParent(argument);

                if (argHead == null)
                    argHead = Token.emptyToken;

                return ImmutableSet.of(
                        "aL+pL|" + argument.lemma + " " + predicate.lemma,
                        "aL+aD+ahL|" + argument.lemma + " " + argument.syntacticHeadRelation + " " + argHead.lemma,
                        "pD|" + predicate.syntacticHeadRelation
                );
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

