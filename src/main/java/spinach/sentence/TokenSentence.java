package spinach.sentence;

import com.google.common.collect.ArrayListMultimap;

import java.util.*;

/**
 * @author Calvin Huang
 *         <p/>
 *         List of tokens that keeps track of each token's child.
 *         Allows for fast computation of multiple token-related features
 *         such as children/descendants of some token.
 *         <p/>
 *         Any methods in this class that return tokens must be called
 *         with tokens that are already part of this sentence, otherwise
 *         undesireable behavior may occur.
 */
public class TokenSentence implements Iterable<Token> {

    protected List<Token> sentenceTokens = new ArrayList<Token>();
    protected ArrayListMultimap<Integer, Token> children = ArrayListMultimap.create();

    /**
     * Find the token at some index in this sentence.
     *
     * @param index index to look for (0-based)
     * @return token at that index
     */
    public Token tokenAt(int index) {
        return sentenceTokens.get(index);
    }

    /**
     * Add a token to the end of the sentence
     *
     * @param token token to be added
     */
    public void addToken(Token token) {
        sentenceTokens.add(token);
        children.put(token.headSentenceIndex, token);
    }

    /**
     * Find the parent of some token
     *
     * @param t token whose parent we are looking for
     * @return syntactic head of t, null if t is root
     */
    public Token getParent(Token t) {
        Token child = sentenceTokens.get(t.sentenceIndex);
        if (child.headSentenceIndex < 0)
            return null;
        return sentenceTokens.get(child.headSentenceIndex);
    }

    /**
     * Get the children of some token
     *
     * @param t token whose children we are looking for
     * @return ordered list of token's children
     */
    public List<Token> getChildren(Token t) {
        return children.get(t.sentenceIndex);
    }

    /**
     * Get the descendants of some token
     *
     * @param t token whose descendants we are looking for
     * @return set of token's descendants
     */
    public Set<Token> getDescendants(Token t) {
        Set<Token> descendants = new HashSet<Token>();
        List<Token> children = getChildren(t);
        descendants.addAll(children);
        for (Token child : children) {
            descendants.addAll(getDescendants(child));
        }

        return descendants;
    }

    /**
     * Get the ancestors of some token
     *
     * @param t token whose ancestors we are looking for
     * @return ordered list of token's ancestors (going head to head)
     */
    public List<Token> getAncestors(Token t) {
        List<Token> ancestors = new ArrayList<Token>();
        Token currToken = t;
        while (currToken.headSentenceIndex >= 0) {
            currToken = getParent(currToken);
            ancestors.add(currToken);
        }

        return ancestors;
    }

    public List<Token> getSiblings(Token t) {
        List<Token> siblings = new ArrayList<Token>();
        Token parent = getParent(t);
        if (parent == null)
            return siblings;
        siblings = getChildren(parent);
        siblings.remove(t);
        return siblings;
    }

    public Deque<Token> getLeftSiblings(Token t) {
        Deque<Token> leftSiblings = new ArrayDeque<Token>();
        for (Token sibling : getSiblings(t))
            if (t.sentenceIndex > sibling.sentenceIndex)
                leftSiblings.add(sibling);

        return leftSiblings;
    }

    public Deque<Token> getRightSiblings(Token t) {
        Deque<Token> rightSiblings = new ArrayDeque<Token>();
        for (Token sibling : getSiblings(t))
            if (t.sentenceIndex < sibling.sentenceIndex)
                rightSiblings.add(sibling);

        return rightSiblings;
    }

    /**
     * Find a common ancestor of two tokens
     *
     * @param a first token
     * @param b second token
     * @return the lowest common ancestor of both tokens
     */
    public Token getCommonAncestor(Token a, Token b) {
        Deque<Token> aAncestors = new ArrayDeque<Token>();
        Deque<Token> bAncestors = new ArrayDeque<Token>();
        aAncestors.add(a);
        aAncestors.addAll(getAncestors(a));
        bAncestors.add(b);
        bAncestors.addAll(getAncestors(b));

        Token commonAncestor = aAncestors.getLast();
        while (!aAncestors.isEmpty() && !bAncestors.isEmpty() && aAncestors.peekLast().equals(bAncestors.peekLast())) {
            commonAncestor = aAncestors.removeLast();
            bAncestors.removeLast();
        }

        return commonAncestor;

    }

    /**
     * Find the path between some token and some ancestor of that token
     *
     * @param a        the beginning token
     * @param ancestor some ancestor of a
     * @return a deque starting from a, going from token to head and ending at ancestor
     */
    public Deque<Token> ancestorPath(Token a, Token ancestor) {
        Deque<Token> path = new ArrayDeque<Token>();
        Token currentToken = a;
        path.add(a);

        while (!currentToken.equals(ancestor)) {
            currentToken = getParent(currentToken);
            if (currentToken == null)
                throw new IllegalArgumentException("Ancestor is not ancestor of provided token");
            path.add(currentToken);
        }

        return path;
    }

    /**
     * The size of the sentence.
     *
     * @return number of tokens in the sentence
     */
    public int size() {
        return sentenceTokens.size();
    }

    /**
     * An iterator through the sentence tokens
     *
     * @return iterator through sentence tokens
     */
    public Iterator<Token> iterator() {
        return sentenceTokens.iterator();
    }
}
