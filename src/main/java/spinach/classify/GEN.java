package spinach.classify;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

/**
 * Interface for a class that takes in a sentence of tokens,
 * and generates a semantic frame set from it
 *
 * @author Calvin Huang
 */
public interface GEN {

    /**
     * Given a sentence of tokens, determines the predicates of that sentence and the arguments of said sentence
     *
     * @param sentence sentence to analyze
     * @return SemanticFrameSet with the original sentence, along with predicates and arguments
     */
    public SemanticFrameSet parse(TokenSentence sentence);

    /**
     * Given a sentence with predicates, determines the arguments of the predicates.
     *
     * @param sentence sentence w/ identified predicates
     * @return sentence w/ identified arguments
     */
    public SemanticFrameSet argParse(TokenSentenceAndPredicates sentence);

    /**
     * Given a sentence, determines the predicates in that sentence.
     *
     * @param sentence sentence to analyze
     * @return sentence w/ identified predicates
     */
    public TokenSentenceAndPredicates predParse(TokenSentence sentence);
}
