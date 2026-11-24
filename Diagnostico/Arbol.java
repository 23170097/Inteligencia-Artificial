public class Arbol {
    Nodo raiz;

    public void insertar(int valor) {
        raiz = insertarRec(raiz, valor);
    }

    private Nodo insertarRec(Nodo actual, int valor) {
        if (actual == null) return new Nodo(valor);
        if (valor < actual.valor) actual.izq = insertarRec(actual.izq, valor);
        else if (valor > actual.valor) actual.der = insertarRec(actual.der, valor);
        return actual;
    }

    public boolean buscar(int valor) {
        Nodo actual = raiz;
        while (actual != null) {
            if (valor == actual.valor) return true;
            actual = (valor < actual.valor) ? actual.izq : actual.der;
        }
        return false;
    }

    public void imprimirInOrden(Nodo actual) {
        if (actual != null) {
            imprimirInOrden(actual.izq);
            System.out.print(actual.valor + " ");
            imprimirInOrden(actual.der);
        }
    }

    public static void main(String[] args) {
        Arbol abb = new Arbol();
        int[] datos = {50, 30, 70, 20, 40, 60, 80};
        for (int n : datos) abb.insertar(n);

        System.out.print("ABB en orden: ");
        abb.imprimirInOrden(abb.raiz);
        System.out.println("\n¿Existe 40? " + abb.buscar(40));
        System.out.println("¿Existe 99? " + abb.buscar(99));
    }
}