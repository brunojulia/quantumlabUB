// AQUEST PROGRAMA FA EL MATEIX QUE L'ALTRE AFEGINT QUE TENIM UNS VAIXELLS EN ALGUNES POSICIONS DE LA QUADRÍCULA, I MIREM SI EN FER UNA MESURA DE LA POSICIÓ DE L'ELECTRÓ
// COINCIDEIX AMB LA DEL VAIXELL. SI ÉS AIXÍ, EL VAIXELL S'ENFONSA

// TAMBÉ AFEGIM EL SEGÜENT: FEM DIFERENTS RONDES, A CADA UNA ES CREEN 100 NOMBRES DISTRIBITS SEGONS LA DENSITAT (QUE SIMULEN LA POSICIÓ DE L'ELECTRÓ I S'ASSOCIA A UNA CASELLA,
// SERIA COM FER 100 MESURES DE COP) I QUAN FEM MÉS RONDES, MÉS PUNTS TENIM. 
// PROBLEMA: NO S'AJUSTA TANT BÉ A LA DISTRIBUCIÓ QUE QUAN FEIEM ELS 10.000 NOMBRES DE COP. HAURÍEM DE GENERAR LLAVORS TOTS ELS NOMBRES
// DE COP I ALHORA DE "FER UNA MESURA" SIMPLEMENT AGAFAR ALGUNS D'AQUESTS NOMBRES I MOSTRAR-LOS PER PANTALLA, I AIXÍ A CADA RONDA? 


using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Random = System.Random;
using Math = System.Math;

public class ProvaMatriusDef : MonoBehaviour
{
        public double L;
        public int ncas;
        public int [] matriu;

    public double densitat(double x, double y)
    {
        // La densitat de probabilitat correspon a la d'un electró tancada en una caixa 2D, al nivell energètic donat
        double funcio;
        double N;
        int nivell;
        N=2.0/L;
        nivell = 1;
        funcio = Math.Pow((N * Math.Sin((nivell*Math.PI*x)/L) * Math.Sin((nivell*Math.PI*y)/L)),2);
        return funcio;
    }
    

    public int[] GenNom(int ndat)
    {
     // Generem 100 nombres distribuïts segons la funció "densitat" (Algoritme de Metropolis)
        //Debug.Log(ndat);
        double [,] xy_nombres = new double [ndat,2]; // vector de ndat nombres, on els ndat nombres tenen 2 components (x,y)
        double [] x0 = new double [2]; double [] chi = new double [2]; double [] q = new double [2] ; 
        double r1; double r2; double r; double variacio;
        int comptador; int iteracio;
        Random _Random = new Random (); 
        int fila; int columna; 
        int [] casella = new int [ndat];

        variacio = 0.00001;
        x0[0] = 2.0; x0[1]=2.0;
        iteracio = 0;
        comptador = 0;

        do 
        {
            // Generem un nombre chi distribuit uniformement entre 0 i 1 (chi té dos components!)
            double chi0 = _Random.NextDouble();
            double chi00 = _Random.NextDouble();
            chi[0] = (1.0-(-1.0))*chi0 + (-1.0);
            chi[1] = (1.0-(-1.0))*chi00 + (-1.0);

            // Definim:
            q[0] = x0[0]+(chi[0]+variacio);
            q[1] = x0[1]+(chi[1]+variacio);
            r1=densitat(q[0], q[1]);
            r2=densitat(x0[0], x0[1]);
            r=r1/r2;
            double p = _Random.NextDouble();

            // Acceptem o no el nou nombre distribuit segons densitat (també s'ha de complir que estigui dins la caixa)
            if(r > p)
            {
                if (q[0]>=0.0 & q[0]<=L & q[1]>=0.0 & q[1]<=L )
                {
                    x0[0]=q[0]; x0[1]=q[1];
                    xy_nombres[comptador,0]=x0[0];
                    xy_nombres[comptador,1]=x0[1]; 
                    //Debug.Log("Nombre: " + x0[0] + " , " + x0[1]);
                    comptador++;
                }     
            }
            iteracio++;
        } while(comptador < ndat);
        //Debug.Log("Comptador final: " + comptador);
        //Debug.Log(xy_nombres[ndat-1,0] + " , " + xy_nombres[ndat-1,1]);


        // Assignarem a cada nombre(x,y), un valor que correspondrà a una casella de les quadrícules.
        // Les caselles seran; 0,1,2,...L, L+1... d'esquerra a dreta i de dalt a baix
        fila=0; columna=0;
        for (int k=0; k<ndat; k++)
        {
            for (int a=0; a<(int)L; a++)
            {
                if (xy_nombres[k,0]>=a & xy_nombres[k,0]<a+1)
                {
                    fila=a;
                    break;
                }
            }
            //Debug.Log("fila: " + fila);
            for (int b=0; b<(int)L; b++)
            {
                if (xy_nombres[k,1]>=b & xy_nombres[k,1]<b+1)
                {
                    columna=b;
                    break;
                }
            }
            //Debug.Log("columna: " + columna);
            // En el vector casella guardem la casella en la qual està cada punt xy_nombres, per ordre.
            casella[k] = fila + (int)L*columna;
            matriu[casella[k]]=matriu[casella[k]]+1;
            //Debug.Log("Electró a la casella: " + casella[k]);
        }
        for (int i=0; i<ncas; i++)
        {
            Debug.Log("CASELLA: " + i + "   PUNTS: " + matriu[i]);
        }
    

        return casella;
    }


// ------------------------------------------------------------------------------------------------------------------

    
    // Start is called before the first frame update
    void Start()
    {
     L=10.0;
     int ndatmesures;
     ndatmesures = 100;
     int[] mesura_electro = new int [ndatmesures];
     int nquad;
     nquad = (int)Math.Pow(L,2);
     ncas = (int)Math.Pow(L,2);
     matriu = new int [ncas];
    
     // Assignem els vaixells en una casella determinada
     int vaixell1; int vaixell2;
     vaixell1=7; vaixell2=10;

     
     // Omplim la quadrícula de "0", i algunes amb "1" que serien en les quals es troben els vaixells
     // A continuació, fem una mesura de la posició de l'electró (generem un nombre aleatori segons la
     // densitat de probabilitat i aquesta serà la posició) i si cau sobre el vaixell l'eliminem
     
     /*int [] quadricula = new int[nquad];
     for (int i=0; i<nquad; i++) 
     {
            quadricula[i]=0;

     }
     //quadricula[vaixell1] = 1;
     //quadricula[vaixell2] = 1;*/

     int rondes;
     rondes=100;

     for (int k=0; k<rondes; k++)
        {
         Debug.Log("Ronda número " + k);
         mesura_electro = GenNom(ndatmesures);
         // Comprovem si l'electró ha caigut sobre algun vaixell
         for (int i=0; i<ndatmesures; i++)
            {
             if (mesura_electro[i] == vaixell1)
                {
                    Debug.Log("Vaixell 1 enfonsat");
                }
             if (mesura_electro[i] == vaixell2)
                {
                    Debug.Log("Vaixell 2 enfonsat");
                }    
            }
        }

    }

    // Update is called once per frame
    void Update()
    {
    }
}



