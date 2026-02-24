#ifndef HVS
#define HVS

#include "hvs.h"
#define DISPLAY_LUMINANCE 60
#define PIXEL_WIDTH 0.000259
#define VIEW_DISTANT 0.5

### python translation of hvs.c from polytech nantes

import numpy as np

DISPLAY_LUMINANCE = 60
PIXEL_WIDTH = 0.000259
VIEW_DISTANT = 0.5


# //Model 2: DCTune Andrew B.Watson.
# //Society for Information Display Digest of Technical Papers XXIV,1993
# //AUTHORS: Andrew B.Watson
# //TITLE: DCTune: A Technique For Visual Optimization of DCT Quantization Matrices for Individual Images
# //!!Take care: Here we don't dip into the Spatial Error models something

# //Pay attension here
# //Usage:
# //use the lm() firstly, then use the cm secondly, Voila, result comes
# //Example:
# //	  lm(y_4x4,DCTune_mask);
# //	  cm(y_4x4,DCTune_mask);

# //The Quantization matrices for image(image-independent)
# extern double DCTune_m_4x4[4][4]={{8,5,12,25},
# 					{7,8,20,34},
# 				    {9,18,34,52},
# 				    {24,39,51,60}};

DCTune_m_4x4 = np.asarray([[  8,  5, 12, 25],
                           [  7,  8, 20, 34],
                           [  9, 18, 34, 52],
                           [ 24, 39, 51, 60]])

DCTune_m_8x8= np.asarray([[ 16, 11, 10, 16, 24, 40, 51, 61],
                          [ 12, 12, 14, 19, 26, 58, 60, 55],
                          [ 14, 13, 16, 24, 40, 57, 69, 56],
                          [ 14, 17, 22, 29, 51, 87, 80, 62],
                          [ 18, 22, 37, 56, 68,109,103, 77],
                          [ 24, 35, 55, 64, 81,104,113, 92],
                          [ 49, 64, 78, 87,103,121,120,101],
                          [ 72, 92, 95, 98,112,100,103, 99])


# //luminance masking function
# //parameter coef: the input 4x4 DCT domain coef values
# //parameter t: the output threshold for 4x4 DCT domain
# extern void lm_4x4(int (*coef)[4],double (*t)[4])

def lm_4x4(coef, t): # in-place

	# double dc;
	# int i,j;

	# dc=(double)abs(coef[0][0])/1024;
    dc = np.abs(coef[0,0])/1024

# 	for(i=0;i<4;i++)
# 		for(j=0;j<4;j++)
# 			t[i][j]=DCTune_m_4x4[i][j]/2*pow(dc,0.649);
    for i in range(4):
        for j in range(4):
            t[i,j] = DCTune_m_4x4[i,j]/2*np.pow(dc,0.649)
# }

# //luminance masking function
# //parameter coef: the input 8x8 DCT domain coef values
# //parameter t: the output threshold for 4x4 DCT domain
# extern void lm_8x8(double **coef,double **t)
# {
	# double dc;
	# int i,j;

	# dc=fabs(coef[0][0])/1024;
    dc = np.abs(coef[0,0])/1024

# //	printf("firts t:\n");
# 	for(i=0;i<8;i++)
# 	{
# 		for(j=0;j<8;j++)
# 		{
# 			t[i][j]=DCTune_m_8x8[i][j]/2*pow(dc,0.649);
# //			printf("%6.2f ",t[i][j]);
# 		};
# //		printf("\n");
# 	};
# }

    for i in range(8):
        for j in range(8):
            t[i,j] = DCTune_m_4x4[i,j]/2*np.pow(dc,0.649)


# //contrast masking function( The full DCTune model)
# //!!Take care: the parameter t here should be after lm()'s caculation.
# //parameter coef: the input 4x4 DCT domain coef values
# //parameter t: the output threshold for 4x4 DCT domain
# extern void cm_4x4(int (*coef)[4],double (*t)[4])
# {
	int i,j;
	double m[4][4];


	for(i=0;i<4;i++)
		for(j=0;j<4;j++)
		{
			if(i==0&&j==0)
				m[i][j]=t[i][j];
			else
				m[i][j]=pow(abs(coef[i][j]),0.7)*pow(fabs(t[i][j]),0.3);
		}

	
	for(i=0;i<4;i++)
		for(j=0;j<4;j++)
			if(m[i][j]>t[i][j])
				t[i][j]=m[i][j];
}

//contrast masking function( The full DCTune model)
//!!Take care: the parameter t here should be after lm()'s caculation.
//parameter coef: the input 8x8 DCT domain coef values
//parameter t: the output threshold for 4x4 DCT domain
extern void cm_8x8(double **coef,double **t)
{
	int i,j;
	double m[8][8];


	//printf("m:\n");
	for(i=0;i<8;i++)
	{
		for(j=0;j<8;j++)
		{
			if(i==0&&j==0)
				m[i][j]=t[i][j];
			else
				m[i][j]=pow(fabs(coef[i][j]),0.7)*pow(fabs(t[i][j]),0.3);
			//printf("%6.2f ",m[i][j]);
		}
		//printf("\n");
	};
	
	for(i=0;i<8;i++)
		for(j=0;j<8;j++)
			if(m[i][j]>t[i][j])
				t[i][j]=m[i][j];

/*	printf("t:\n");
	for(i=0;i<8;i++)
	{
		for(j=0;j<8;j++)
		{
			printf("%6.2f ",t[i][j]);
		};
		printf("\n");
	};*/
}

void operator_hvs(double **dct_8x8,double **hvs_8x8)
{
	int i,j;
	
	for(i=0;i<8;i++)
		for(j=0;j<8;j++)
		{
			if(dct_8x8[i][j]<0)
				hvs_8x8[i][j]=fabs(hvs_8x8[i][j])*-1;
			else
				hvs_8x8[i][j]=fabs(hvs_8x8[i][j])*1;
			//if(fabs(dct_8x8[i][j])<fabs(hvs_8x8[i][j]))
			//	hvs_8x8[i][j]=0;
		}
};

//Model 3:Jnd Model by X.H.Zhang
//Journal of visual Communication &Image Representation., June 2007.
//Xiaohui Zhang, Weisi Lin, Ping Xue
//Just-noticeable difference estimation with pixels in images

//Pay attension here
//Usage:
//use the gen_dct_8x8_csf_adl() firstly, use the gen_com_dct_8x8(scf,dc) secondly, use the block_class() thirdly, final step XHZ_Jnd_model(), Voila, result come out 
//Example:
//    gen_dct_8x8_csf(s_csf);
//	  gen_com_dct_8x8_csf_adl(s_csf,y_4x4[0][0]);
//	  block_class_8x8(y_4x4,block_c,s_csf);
//	  XHZ_Jnd_model_8x8(xhz_mask,block_c,s_csf);

//DCT Normalizaing factor
//parameter x: position or frequency of DCT coef., only ditinguish 0 or non-0
//parameter N: NxN DCT block
extern double phi(int x,int N)
{
	double value;
	if(x==0)
	{
		value=sqrt(1/(double)N);
		return value;
	}
	else
	{
		value=sqrt(2/(double)N);
		return value;
	}
}

// The minimum luminance threshold
// parameter L: background luminance
extern double T_min(double L)
{
	if(L<=13.45)
		return 0.142*pow(L/13.45,0.649);
	else
		return L/94.7;
}

// Spatail frequency
// parameter L: background luminace
extern double sf(double L)
{
	if(L<=300)
		return 6.78*pow(L/300,0.182);
	else
		return 6.78;
}

//visual angles of a pixel
//parameter lambda_u: display width of a pixel
//parameter lambda: viewing distance
extern double omega(double lambda_u,double lambda)
{
	return 2*atan(lambda_u/lambda/2);
}

//Spatial frequency, asscociated with i,jth basis function( from Peteson's model) 
//parameter N: NxN DCT block
//parameter i,j: the i,j(th) DCT coef.
//parameter x_omega,y_omega: the x and y axis of pixel viusal angles
extern double sf_xy(int N,int i, int j, double x_omega, double y_omega)
{
	return 0.5/(double)N*sqrt(i*i/pow(x_omega,2)+j*j/pow(y_omega,2));
}

//angular parameter
//parameter N: NxN DCT block
//parameter f_i_0,f_0_j,f_i_j: the spatail frequency asscociated with i_0,0_j,i_j th basis function
extern double theta(int N,double f_i_0,double f_0_j, double f_i_j)
{
	return 2*f_i_0*f_0_j/pow(f_i_j,2);
}

//steepness of the parabola
//parameter L: background luminance
extern double K(double L)
{
	if(L<=300)
		return 3.125*pow(L/300,0.0706);
	else
		return 3.125;
}


//Spatial CSF effect
//!!Take care: this functions need the above all functions.
//parameter i,j:the i,j(th) DCT coef.
//parameter G: total number of gray levels
//parameter Lmax,Lmin: the display luminances corresponding to maximum and minimum gray levels(i.e,255 and 0)
//parameter N: NxN DCT block
//parameter L: background luminance
//parameter s: the spatial summation effect
//parameter r: the oblique effect
//parameter lambda_u: display width of a pixel
//parameter lambda: viewing distance
extern double T(int i, int j,int G,int Lmax,int Lmin,int N,double L,double s,double r,double lambda_u,double lambda)
{
   double T_0_ij;
   double f_i_j;
   double f_i_0;
   double f_0_j;
   double f_p;
   double theta_i_j;
   double k;
   double phi_i,phi_j;
   double T_part1;
   double T_part2;
   double omega_x;
   double omega_y;

	 //!!!!important
	 //the paper is not carefully point out which kind of angel value they used
	 //after experiments, we point out the it should not be the roations, so ...balalalalala
   omega_x=omega(lambda_u,lambda)/3.14*180;
   omega_y=omega(lambda_u,lambda)/3.14*180;

   f_i_j=sf_xy(N,i,j,omega_x,omega_y);
   f_i_0=sf_xy(N,i,0,omega_x,omega_y);
   f_0_j=sf_xy(N,0,j,omega_x,omega_y);
   f_p=sf(L);
   k=K(L);
   theta_i_j=theta(N,f_i_0,f_0_j,f_i_j);
   T_part1=log(s*T_min(L)/(r+(1-r)*(1-pow(theta_i_j,2))));
   T_part2=k*pow((log(f_i_j)-log(f_p)),2);
   T_0_ij=T_part1+T_part2;
   //T_0_ij=pow(2.71828182845904523536,T_0_ij);
   T_0_ij=pow(2,T_0_ij);
   phi_i=phi(i,N);
   phi_j=phi(j,N);

   return (double)G/(phi_i*phi_j*(Lmax-Lmin))*T_0_ij;
}

//Generate DCT_4x4_Spatial_CSF_function
//!!Take care: it bases on the T() funciton above, caculating every coef.
//paramter s_csf: the output, the spatial_csf threshold value
extern void gen_dct_4x4_csf(double (*s_csf)[4])
{
	int i,j;
	for(i=0;i<4;i++)
		for(j=0;j<4;j++)
			if(i!=0||j!=0)
			s_csf[i][j]=T(i,j,256,100,1,4,DISPLAY_LUMINANCE,0.25,0.6,PIXEL_WIDTH,VIEW_DISTANT);
	s_csf[0][0]=min(s_csf[1][0],s_csf[0][1]);
}

//Generate DCT_8x8_Spatial_CSF_function
//!!Take care: it bases on the T() funciton above, caculating every coef.
//paramter s_csf: the output, the spatial_csf threshold value
extern void gen_dct_8x8_csf(double **s_csf)
{
	int i,j;
	for(i=0;i<8;i++)
		for(j=0;j<8;j++)
			if(i!=0||j!=0)
			s_csf[i][j]=T(i,j,256,100,1,8,DISPLAY_LUMINANCE,0.25,0.6,PIXEL_WIDTH,VIEW_DISTANT);
	s_csf[0][0]=min(s_csf[1][0],s_csf[0][1]);
}

//Adjustment for luminance adaption
//parameter N: NxN DCT block
//parameter k1,k2: some values from experiments make the curve more reasonable
//parameter dc: you know dc, don't you? if you don't know....en......no words
//parameter G: total number of gray levels
//parameter lambda1,lambda2: the same as k1,k2
extern double adjust_l(double N,double k1,double k2, double dc,double G,double lambda1,double lambda2)
{
	if(dc<(G*N/2))
		return k1*pow((1-2*dc/(G*N)),lambda1)+1;
	else
		return k2*pow((2*dc/(G*N)-1),lambda2)+1;
}

//Combination of CSF and Adjustment for luminance adaption
//!!Take care: it use the adjust_l funciton
//parameter s_csf: the output of s_scf after processing by the Adjustment for luminace adapation
//parameter dc: ok, i tell you, the 0-frequency of DCT coefficient
extern void gen_com_dct_4x4_csf_adl(double (*s_csf)[4],double dc)
{
	double adl;
	int i,j;

	adl=adjust_l(4,2,0.8,dc,255,3,2);

	for(i=0;i<4;i++)
		for(j=0;j<4;j++)
			s_csf[i][j]=s_csf[i][j]*adl;
}

extern void gen_com_dct_8x8_csf_adl(double **s_csf,double dc)
{
	double adl;
	int i,j;

	adl=adjust_l(8,2,0.8,dc,255,3,2);

	for(i=0;i<8;i++)
		for(j=0;j<8;j++)
			s_csf[i][j]=s_csf[i][j]*adl;
}


//Block classification for contrast masking 4x4 block
//!!Take care: this funtion is a little complex, maybe you need to pay more attention to papers, my recommendation is a paper called "A Perceptual model for JPEG application based on block classification,texture masking, and luminance masking", doneby Tong and somebodys.
//parameter coef: the 4x4 dct coef.
//parameter block_c: the ouput block classification information
//parameter s_csf: after Spatial CSF, after lum_adp, then it is
extern int block_class_4x4(int (*coef)[4],double (*block_c)[4],double (*s_csf)[4])
{
	int i,j;

	double dc;
	double lf,mlf;
	double mf,mmf;
	double hf,mhf;
	double TexE,E1,E2;
	double zeta;

	int mode;// 1:PLAIN,2:EDGE,3:TEXURE.

	dc=abs(coef[0][0]);
	lf=abs(coef[0][1])+abs(coef[1][0]);
	mf=abs(coef[0][2])+abs(coef[2][0])+abs(coef[1][1])+abs(coef[0][3])+abs(coef[3][0]);
	hf=abs(coef[2][1])+abs(coef[1][2])+abs(coef[1][3])+abs(coef[2][2])+abs(coef[2][3])+abs(coef[3][1])+abs(coef[3][2])+abs(coef[3][3]);

	mlf=lf/2;
	mmf=mf/5;
	mhf=hf/8;

	//some try
	dc=dc/2;
	lf=lf/2*5/2;
	mf=mf/2*12/5;
	hf=hf/2*52/8;
	mlf=mlf/2;
	mmf=mmf/2;
	mhf=mhf/2;

	TexE=mf+hf;
	E1=(mlf+mmf)/mhf;
	E2=mlf/mmf;

	if(TexE<=125)
	{
		mode=1;
		zeta=1;
	}
	else
		if(TexE<=290)
		{
			if(E1>16||(max(E1,E2)>=7&&min(E1,E2)>=5))
			{
				mode=2;
				if((lf+mf)>400)
					zeta=1.25;
				else
					zeta=1.125;
			}
			else
			{
				mode=1;
				zeta=1;
			}
		}
		else
			if(TexE<=900)
			{
				if(E1>16||(max(E1,E2)>=7&&min(E1,E2)>=5))
				{
					mode=2;
					if((lf+mf)>400)
						zeta=1.25;
					else
						zeta=1.125;
				}
				else
				{
					mode=3;
					zeta=1+(TexE-290)/(1800-290)*1.25;
				}
			}
			else
				{
					if(E1>16||(max(E1,E2)>=0.7&&min(E1,E2)>=0.5))
					{
						mode=2;
						if((lf+mf)>400)
							zeta=1.25;
						else
							zeta=1.125;
					}
					else
					{
						mode=3;
						zeta=1+(TexE-290)/(1800-290)*1.25;
					}
				}
	for(i=0;i<4;i++)
		for(j=0;j<4;j++)
		{
			double value=pow(fabs((double)coef[i][j]/s_csf[i][j]),0.36);
			block_c[i][j]=zeta*max(1,value);
		}
	if(mode==2)
	{
		block_c[0][1]=zeta;
		block_c[0][2]=zeta;
		block_c[0][3]=zeta;
		block_c[1][0]=zeta;
		block_c[1][1]=zeta;
		block_c[2][0]=zeta;
		block_c[3][0]=zeta;
	}
	return mode;
}


extern double block_freq[8][8]={{0,1,1,2,2,2,2,3},
								{1,1,2,3,3,3,3,3},
								{1,2,2,3,3,3,3,3},
								{2,3,3,2,3,3,3,3},
								{2,3,3,3,3,3,3,3},
								{2,3,3,3,3,3,3,3},
								{2,3,3,3,3,3,3,3},
								{3,3,3,3,3,3,3,3}};


//Block classification for contrast masking 4x4 block
//!!Take care: this funtion is a little complex, maybe you need to pay more attention to papers, my recommendation is a paper called "A Perceptual model for JPEG application based on block classification,texture masking, and luminance masking", doneby Tong and somebodys.
//parameter coef: the 4x4 dct coef.
//parameter block_c: the ouput block classification information
//parameter s_csf: after Spatial CSF, after lum_adp, then it is
extern int block_class_8x8(double **coef,double **block_c,double **s_csf)
{
	int i,j;

	double dc,dc_num;
	double lf,mlf,lf_num;
	double mf,mmf,mf_num;
	double hf,mhf,hf_num;
	double TexE,E1,E2;
	double zeta;

	int mode;// 1:PLAIN,2:EDGE,3:TEXURE.

	dc=0;
	lf=0;
	mf=0;
	hf=0;
	dc_num=0;
	lf_num=0;
	mf_num=0;
	hf_num=0;

	for(i=0;i<8;i++)
		for(j=0;j<8;j++)
		{
			if(block_freq[i][j]==0)
			{
				dc+=fabs(coef[i][j]);
				dc_num++;
			}
			else
				if(block_freq[i][j]==1)
				{
					lf+=fabs(coef[i][j]);
					lf_num++;
				}
				else
					if(block_freq[i][j]==2)
					{
						mf+=fabs(coef[i][j]);
						mf_num++;
					}
					else
					{
						hf+=fabs(coef[i][j]);
						hf_num++;
					}
		}

	mlf=lf/lf_num;
	mmf=mf/mf_num;
	mhf=hf/hf_num;

	TexE=mf+hf;
	E1=(mlf+mmf)/mhf;
	E2=mlf/mmf;

	if(TexE<=125)
	{
		mode=1;
		zeta=1;
	}
	else
		if(TexE<=290)
		{
			if(E1>16||(max(E1,E2)>=7&&min(E1,E2)>=5))
			{
				mode=2;
				if((lf+mf)>400)
					zeta=1.25;
				else
					zeta=1.125;
			}
			else
			{
				mode=1;
				zeta=1;
			}
		}
		else
			if(TexE<=900)
			{
				if(E1>16||(max(E1,E2)>=7&&min(E1,E2)>=5))
				{
					mode=2;
					if((lf+mf)>400)
						zeta=1.25;
					else
						zeta=1.125;
				}
				else
				{
					mode=3;
					zeta=1+(TexE-290)/(1800-290)*1.25;
				}
			}
			else
				{
					if(E1>16||(max(E1,E2)>=0.7&&min(E1,E2)>=0.5))
					{
						mode=2;
						if((lf+mf)>400)
							zeta=1.25;
						else
							zeta=1.125;
					}
					else
					{
						mode=3;
						zeta=1+(TexE-290)/(1800-290)*1.25;
					}
				}
	for(i=0;i<8;i++)
		for(j=0;j<8;j++)
		{
			double value=pow(fabs(coef[i][j]/s_csf[i][j]),0.36);
			block_c[i][j]=zeta*max(1,value);
		}
	if(mode==2)
	{
		for(i=0;i<8;i++)
			for(j=0;j<8;j++)
			{
				if(block_freq[i][j]==1||block_freq[i][j]==2)
					block_c[i][j]=zeta;
			}
	}
	return mode;
}

#endif

