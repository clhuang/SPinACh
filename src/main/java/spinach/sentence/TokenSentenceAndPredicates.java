package spinach.sentence;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TokenSentenceAndPredicates extends TokenSentence {
    protected List<Token> predicateList = new ArrayList<Token>();

    public TokenSentenceAndPredicates() {
    }

    public TokenSentenceAndPredicates(TokenSentence sentence) {
        sentenceTokens = sentence.sentenceTokens;
        children = sentence.children;
    }

    /**
     * Add a predicate to the list of predicates.
     *
     * @param predicate predicate to be added
     */
    public void addPredicate(Token predicate) {
        predicateList.add(predicate);
    }

    /**
     * Add a bunch of predicates to the list of predicates.
     * The predicates should be in order.
     *
     * @param predicateList list of predicates to be appended
     */
    public void addPredicates(List<Token> predicateList) {
        predicateList.addAll(predicateList);
    }

    /**
     * Returns an immutable copy of the list of predicates.
     *
     * @return list of predicates
     */
    public List<Token> getPredicateList() {
        return Collections.unmodifiableList(predicateList);
    }

    /**
     * Returns if a token in this sentence is a predicate
     *
     * @param t token in question
     * @return if t is a predicate
     */
    public boolean isPredicate(Token t){
        return predicateList.contains(t);
    }
}