package spinach.sentence;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableSet;

import java.util.*;

/**
 * List of tokens that keeps track of syntactic dependencies between tokens.
 * Allows for fast computation of multiple token-related features
 * such as children/descendants of some token.
 * <p/>
 * Any methods in this class that return tokens must be called with tokens
 * that are in this sentence, otherwise undesirable behavior will occur.
 *
 * @author Calvin Huang
 */
public class TokenSentence implements Iterable<Token> {

    List<Token> sentenceTokens = new ArrayList<Token>();
    Token root;
    ArrayListMultimap<Integer, Token> children = ArrayListMultimap.create();

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
     * Adds a token to the end of the sentence.
     *
     * @param token token to be added
     */
    public void addToken(Token token) {
        sentenceTokens.add(token);
        if (token.headSentenceIndex >= 0)
            children.put(token.headSentenceIndex, token);
        else
            root = token;
    }

    /**
     * Returns the sentence root.
     *
     * @return sentence root
     */
    public Token getRoot() {
        return root;
    }

    /**
     * Get the parent of some token.
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
     * Get the children of some token.
     *
     * @param t token whose children we are looking for
     * @return ordered list of token's children
     */
    public List<Token> getChildren(Token t) {
        return children.get(t.sentenceIndex);
    }

    /**
     * Get the syntactic descendants of some token.
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
     * Get the ancestors of some token in the syntactic tree.
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

    /**
     * Get the siblings (children of parent) of some token, including that token.
     *
     * @param t token to analyze
     * @return deque of siblings in order
     */
    public Deque<Token> getSiblings(Token t) {
        Deque<Token> siblings = new ArrayDeque<Token>();
        Token parent = getParent(t);
        if (parent == null) {
            siblings.add(t);
            return siblings;
        }
        siblings.addAll(getChildren(parent));
        return siblings;
    }

    /**
     * Get the siblings that appear before some token, inclusive.
     *
     * @param t token to analyze
     * @return deque of preceding siblings in order
     */
    public Deque<Token> getLeftSiblings(Token t) {
        Deque<Token> leftSiblings = new ArrayDeque<Token>();
        for (Token sibling : getSiblings(t)) {
            if (t.sentenceIndex >= sibling.sentenceIndex)
                leftSiblings.add(sibling);
            else
                break;
        }

        return leftSiblings;
    }

    /**
     * Get the siblings that appear after some token, inclusive.
     *
     * @param t token to analyze
     * @return deque of succeeding siblings in order
     */
    public Deque<Token> getRightSiblings(Token t) {
        Deque<Token> rightSiblings = new ArrayDeque<Token>();
        for (Token sibling : getSiblings(t))
            if (t.sentenceIndex <= sibling.sentenceIndex)
                rightSiblings.add(sibling);

        return rightSiblings;
    }

    /**
     * Find a common ancestor of two tokens in the syntactic tree.
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
     * Find the path along the syntactic tree between some token and some ancestor of that token
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
                break;
            path.add(currentToken);
        }

        return path;
    }

    /**
     * Returns the shortest path along the syntactic tree from token a to b.
     *
     * @param a starting token
     * @param b ending token
     * @return deque of tokens along syntactic path from a to b, inclusive
     */
    public Deque<Token> syntacticPath(Token a, Token b) {
        Deque<Token> path = new ArrayDeque<Token>();
        Token ancestor = getCommonAncestor(a, b);
        path.addAll(ancestorPath(a, ancestor));

        Deque<Token> bPath = ancestorPath(b, ancestor);
        bPath.removeLast();
        while (!bPath.isEmpty())
            path.add(bPath.removeLast());

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
    @Override
    public Iterator<Token> iterator() {
        return sentenceTokens.iterator();
    }

    /**
     * Returns the voice of a particular word (active, passive, etc.)
     *
     * @param t token to be analyzed
     * @return the voice of the word
     */
    public String voiceOf(Token t) {
        if (!t.pos.startsWith("VB"))
            return "notVerb";

        int verbContextRightBoundary = t.sentenceIndex - 1;
        int verbContextLeftBoundary = verbContextRightBoundary;

        while (verbContextLeftBoundary > 0 && !tokenAt(verbContextLeftBoundary).pos.equals("CC"))
            verbContextLeftBoundary--;
        if (verbContextLeftBoundary > 0)
            verbContextLeftBoundary--;
        else
            verbContextLeftBoundary = 0;

        Token verbModifier = null;

        for (int i = verbContextRightBoundary; i >= verbContextLeftBoundary; i--) {
            String pos = tokenAt(i).pos;
            if (pos.startsWith("TO") || pos.startsWith("MD") || pos.startsWith("VB") || pos.startsWith("AUX")) {
                verbModifier = tokenAt(i);
                break;
            }
        }

        if (t.pos.equals("VBG") && verbModifier == null)
            return "gerund";
        if (t.pos.equals("VB") && verbModifier != null && verbModifier.pos.equals("TO"))
            return "infinitive";
        if (isBeVerb(t))
            return "copulative";
        if ((t.pos.equals("VBN") || t.pos.equals("VBD")) && verbModifier != null &&
                (isBeVerb(verbModifier) || isGetVerb(verbModifier)))
            return "passive";
        return "active";
    }

    private static final Set<String> getVerbForms = ImmutableSet.of(
            "get", "got", "gotten", "getting", "geting", "gets"
    );

    private static final Set<String> beVerbForms = ImmutableSet.of(
            "be", "am", "is", "was", "are", "were", "been", "being"
    );

    private static boolean isBeVerb(Token t) {
        return beVerbForms.contains(t.form.toLowerCase());
    }

    private static boolean isGetVerb(Token t) {
        return getVerbForms.contains(t.form.toLowerCase());
    }
}
