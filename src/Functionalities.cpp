
#pragma once
#include "Functionalities.h"
#include <algorithm>    // std::rotate
#include <thread>
using namespace std;



/******************************** Functionalities 2PC ********************************/
// Share Truncation, truncate shares of a by power (in place) (power is logarithmic)
void funcTruncate2PC(vector<myType> &a, size_t power, size_t size, size_t party_1, size_t party_2)
{
	assert((partyNum == party_1 or partyNum == party_2) && "Truncate called by spurious parties");

	if (partyNum == party_1)
		for (size_t i = 0; i < size; ++i)
			a[i] = static_cast<uint64_t>(static_cast<int64_t>(a[i]) >> power);

	if (partyNum == party_2)
		for (size_t i = 0; i < size; ++i)
			a[i] = - static_cast<uint64_t>(static_cast<int64_t>(- a[i]) >> power);
}


// XOR shares with a public bit into output.
void funcXORModuloOdd2PC(vector<smallType> &bit, vector<myType> &shares, vector<myType> &output, size_t size)
{
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (bit[i] == 1)
				output[i] = subtractModuloOdd<smallType, myType>(1, shares[i]);
			else
				output[i] = shares[i];
		}
	}

	if (partyNum == PARTY_B)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (bit[i] == 1)
				output[i] = subtractModuloOdd<smallType, myType>(0, shares[i]);
			else
				output[i] = shares[i];
		}
	}
}

void funcReconstruct2PC(const vector<myType> &a, size_t size, string str)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Reconstruct called by spurious parties");

	vector<myType> temp(size);
	if (partyNum == PARTY_B)
		sendVector<myType>(a, PARTY_A, size);

	if (partyNum == PARTY_A)
	{
		receiveVector<myType>(temp, PARTY_B, size);
		addVectors<myType>(temp, a, temp, size);
	
		cout << str << ": ";
		for (size_t i = 0; i < size; ++i)
			print_linear(temp[i], DEBUG_PRINT);
		cout << endl;
	}
}


void funcReconstructBit2PC(const vector<smallType> &a, size_t size, string str)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Reconstruct called by spurious parties");

	vector<smallType> temp(size);
	if (partyNum == PARTY_B)
		sendVector<smallType>(a, PARTY_A, size);

	if (partyNum == PARTY_A)
	{
		receiveVector<smallType>(temp, PARTY_B, size);
		XORVectors(temp, a, temp, size);
	
		cout << str << ": ";
		for (size_t i = 0; i < size; ++i)
			cout << (int)temp[i] << " ";
		cout << endl;
	}
}


void funcConditionalSet2PC(const vector<myType> &a, const vector<myType> &b, vector<smallType> &c, 
					vector<myType> &u, vector<myType> &v, size_t size)
{
	assert((partyNum == PARTY_C or partyNum == PARTY_D) && "ConditionalSet called by spurious parties");

	for (size_t i = 0; i < size; ++i)
	{
		if (c[i] == 0)
		{
			u[i] = a[i];
			v[i] = b[i];
		}
		else
		{
			u[i] = b[i];
			v[i] = a[i];
		}
	}
}


/******************************** Functionalities MPC ********************************/

// Matrix Multiplication of a*b = c with transpose flags for a,b.
// Output is a share between PARTY_A and PARTY_B.
// a^transpose_a is rows*common_dim and b^transpose_b is common_dim*columns
void funcMatMulMPC(const vector<myType> &a, const vector<myType> &b, vector<myType> &c, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b)
{
	log_print("funcMatMulMPC");
#if (LOG_DEBUG)
	cout << "Rows, Common_dim, Columns: " << rows << "x" << common_dim << "x" << columns << endl;
#endif

	if (FOUR_PC)
	{
		size_t size = rows*columns;
		vector<myType> temp(size);
		vector<myType> a_temp = a;
		vector<myType> b_temp = b;
		getVectorfromPrimary<myType>(a_temp, rows*common_dim, "AS-IS", "NATURAL");
		getVectorfromPrimary<myType>(b_temp, common_dim*columns, "AS-IS", "UNNATURAL");

		matrixMultEigen(a_temp, b_temp, c, rows, common_dim, columns, transpose_a, transpose_b);

		if (NON_PRIMARY)
		{
			populateRandomVector<myType>(temp, size, "COMMON", "NEGATIVE");
			addVectors<myType>(c, temp, c, size);
			sendVector<myType>(c, partner(partyNum), size);
		}

		if (PRIMARY)
		{
			receiveVector<myType>(temp, partner(partyNum), size);
			addVectors<myType>(c, temp, c, size);
			funcTruncate2PC(c, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
		}
	}

	if (THREE_PC)
	{
		// cout << "Here" << endl;
		size_t size = rows*columns;
		size_t size_left = rows*common_dim;
		size_t size_right = common_dim*columns;
		vector<myType> A(size_left, 0), B(size_right, 0), C(size, 0);

		if (HELPER)
		{
			vector<myType> A1(size_left, 0), A2(size_left, 0), 
						   B1(size_right, 0), B2(size_right, 0), 
						   C1(size, 0), C2(size, 0);

			populateRandomVector<myType>(A1, size_left, "a_1", "POSITIVE");
			populateRandomVector<myType>(A2, size_left, "a_2", "POSITIVE");
			populateRandomVector<myType>(B1, size_right, "b_1", "POSITIVE");
			populateRandomVector<myType>(B2, size_right, "b_2", "POSITIVE");
			populateRandomVector<myType>(C1, size, "c_1", "POSITIVE");

			addVectors<myType>(A1, A2, A, size_left);
			addVectors<myType>(B1, B2, B, size_right);

			matrixMultEigen(A, B, C, rows, common_dim, columns, 0, 0);
			subtractVectors<myType>(C, C1, C2, size);

			// splitIntoShares(C, C1, C2, size);

			// sendThreeVectors<myType>(A1, B1, C1, PARTY_A, size_left, size_right, size);
			// sendThreeVectors<myType>(A2, B2, C2, PARTY_B, size_left, size_right, size);
			sendVector<myType>(C2, PARTY_B, size);
		}

		if (PRIMARY)
		{
			vector<myType> E(size_left), F(size_right);
			vector<myType> temp_E(size_left), temp_F(size_right);
			vector<myType> temp_c(size);

			if (partyNum == PARTY_A)
			{
				populateRandomVector<myType>(A, size_left, "a_1", "POSITIVE");
				populateRandomVector<myType>(B, size_right, "b_1", "POSITIVE");
				populateRandomVector<myType>(C, size, "c_1", "POSITIVE");
			}

			if (partyNum == PARTY_B)
			{
				populateRandomVector<myType>(A, size_left, "a_2", "POSITIVE");
				populateRandomVector<myType>(B, size_right, "b_2", "POSITIVE");
				receiveVector<myType>(C, PARTY_C, size);
			}			

			// receiveThreeVectors<myType>(A, B, C, PARTY_C, size_left, size_right, size);
			subtractVectors<myType>(a, A, E, size_left);
			subtractVectors<myType>(b, B, F, size_right);


			thread *threads = new thread[2];

			threads[0] = thread(sendTwoVectors<myType>, ref(E), ref(F), adversary(partyNum), size_left, size_right);
			threads[1] = thread(receiveTwoVectors<myType>, ref(temp_E), ref(temp_F), adversary(partyNum), size_left, size_right);
	
			for (int i = 0; i < 2; i++)
				threads[i].join();

			delete[] threads;

			//HEREEEEEEE
			// if (partyNum == PARTY_A)
			// 	sendTwoVectors<myType>(E, F, adversary(partyNum), size_left, size_right);
			// else
			// 	receiveTwoVectors<myType>(temp_E, temp_F, adversary(partyNum), size_left, size_right);	

			// if (partyNum == PARTY_B)
			// 	sendTwoVectors<myType>(E, F, adversary(partyNum), size_left, size_right);
			// else
			// 	receiveTwoVectors<myType>(temp_E, temp_F, adversary(partyNum), size_left, size_right);	
			

			// sendTwoVectors<myType>(E, F, adversary(partyNum), size_left, size_right);
			// receiveTwoVectors<myType>(temp_E, temp_F, adversary(partyNum), size_left, size_right);

			addVectors<myType>(E, temp_E, E, size_left);
			addVectors<myType>(F, temp_F, F, size_right);

			matrixMultEigen(a, F, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(E, b, temp_c, rows, common_dim, columns, 0, 0);

			addVectors<myType>(c, temp_c, c, size);
			addVectors<myType>(c, C, c, size);

			if (partyNum == PARTY_A)
			{
				matrixMultEigen(E, F, temp_c, rows, common_dim, columns, 0, 0);
				subtractVectors<myType>(c, temp_c, c, size);
			}

			funcTruncate2PC(c, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
		}
	}
}


void funcDotProductMPC(const vector<myType> &a, const vector<myType> &b, 
						   vector<myType> &c, size_t size) 
{
	log_print("funcDotProductMPC");

	if (FOUR_PC)
	{
		vector<myType> temp(size);
		vector<myType> a_temp = a;
		vector<myType> b_temp = b;
		getVectorfromPrimary<myType>(a_temp, size, "AS-IS", "NATURAL");
		getVectorfromPrimary<myType>(b_temp, size, "AS-IS", "UNNATURAL");

		for (size_t i = 0; i < size; ++i)
			c[i] = a_temp[i] * b_temp[i];

		if (NON_PRIMARY)
		{
			populateRandomVector<myType>(temp, size, "COMMON", "NEGATIVE");
			addVectors<myType>(c, temp, c, size);
			sendVector<myType>(c, partner(partyNum), size);
		}

		if (PRIMARY)
		{
			receiveVector<myType>(temp, partner(partyNum), size);
			addVectors<myType>(c, temp, c, size);
			funcTruncate2PC(c, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
		}
	}

	if (THREE_PC)
	{
		vector<myType> A(size, 0), B(size, 0), C(size, 0);

		if (HELPER)
		{
			vector<myType> A1(size, 0), A2(size, 0), 
						   B1(size, 0), B2(size, 0), 
						   C1(size, 0), C2(size, 0);

			populateRandomVector<myType>(A1, size, "a_1", "POSITIVE");
			populateRandomVector<myType>(A2, size, "a_2", "POSITIVE");
			populateRandomVector<myType>(B1, size, "b_1", "POSITIVE");
			populateRandomVector<myType>(B2, size, "b_2", "POSITIVE");
			populateRandomVector<myType>(C1, size, "c_1", "POSITIVE");

			// populateRandomVector<myType>(A1, size, "INDEP", "POSITIVE");
			// populateRandomVector<myType>(A2, size, "INDEP", "POSITIVE");
			// populateRandomVector<myType>(B1, size, "INDEP", "POSITIVE");
			// populateRandomVector<myType>(B2, size, "INDEP", "POSITIVE");

			addVectors<myType>(A1, A2, A, size);
			addVectors<myType>(B1, B2, B, size);

			for (size_t i = 0; i < size; ++i)
				C[i] = A[i] * B[i];

			// splitIntoShares(C, C1, C2, size);
			subtractVectors<myType>(C, C1, C2, size);
			sendVector<myType>(C2, PARTY_B, size);

			// sendThreeVectors<myType>(A1, B1, C1, PARTY_A, size, size, size);
			// sendThreeVectors<myType>(A2, B2, C2, PARTY_B, size, size, size);
		}

		if (PRIMARY)
		{
			if (partyNum == PARTY_A)
			{
				populateRandomVector<myType>(A, size, "a_1", "POSITIVE");
				populateRandomVector<myType>(B, size, "b_1", "POSITIVE");
				populateRandomVector<myType>(C, size, "c_1", "POSITIVE");
			}

			if (partyNum == PARTY_B)
			{
				populateRandomVector<myType>(A, size, "a_2", "POSITIVE");
				populateRandomVector<myType>(B, size, "b_2", "POSITIVE");
				receiveVector<myType>(C, PARTY_C, size);
			}			

			// receiveThreeVectors<myType>(A, B, C, PARTY_C, size, size, size);
			vector<myType> E(size), F(size), temp_E(size), temp_F(size);
			myType temp;

			subtractVectors<myType>(a, A, E, size);
			subtractVectors<myType>(b, B, F, size);

			thread *threads = new thread[2];

			threads[0] = thread(sendTwoVectors<myType>, ref(E), ref(F), adversary(partyNum), size, size);
			threads[1] = thread(receiveTwoVectors<myType>, ref(temp_E), ref(temp_F), adversary(partyNum), size, size);
	
			for (int i = 0; i < 2; i++)
				threads[i].join();

			delete[] threads;

			//HEREEEEEEE
			// if (partyNum == PARTY_A)
			// 	sendTwoVectors<myType>(E, F, adversary(partyNum), size, size);
			// else
			// 	receiveTwoVectors<myType>(temp_E, temp_F, adversary(partyNum), size, size);

			// if (partyNum == PARTY_B)
			// 	sendTwoVectors<myType>(E, F, adversary(partyNum), size, size);
			// else
			// 	receiveTwoVectors<myType>(temp_E, temp_F, adversary(partyNum), size, size);

			// sendTwoVectors<myType>(E, F, adversary(partyNum), size, size);
			// receiveTwoVectors<myType>(temp_E, temp_F, adversary(partyNum), size, size);

			addVectors<myType>(E, temp_E, E, size);
			addVectors<myType>(F, temp_F, F, size);

			for (size_t i = 0; i < size; ++i)
			{
				c[i] = a[i] * F[i];
				temp = E[i] * b[i];
				c[i] = c[i] + temp;

				if (partyNum == PARTY_A)
				{
					temp = E[i] * F[i];
					c[i] = c[i] - temp;
				}
			}
			addVectors<myType>(c, C, c, size);
			funcTruncate2PC(c, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
		}
	}
}

//Thread function for parallel private compare
void parallelPC(smallType* c, size_t start, size_t end, int t, 
				const smallType* share_m, const myType* r, 
				const smallType* beta, const smallType* betaPrime, size_t dim)
{
	size_t index3, index2;
	size_t PARTY;

	smallType bit_r, a, tempM;
	myType valueX;

	thread_local int shuffle_counter = 0;
	thread_local int nonZero_counter = 0;

	//Check the security of the first if condition
	for (size_t index2 = start; index2 < end; ++index2)
	{
		if (beta[index2] == 1 and r[index2] != MINUS_ONE)
			valueX = r[index2] + 1;
		else
			valueX = r[index2];

		if (beta[index2] == 1 and r[index2] == MINUS_ONE)
		{
			//One share of zero and other shares of 1
			//Then multiply and shuffle
			for (size_t k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				c[index3] = aes_common->randModPrime();
				if (partyNum == PARTY_A)
					c[index3] = subtractModPrime((k!=0), c[index3]);

				c[index3] = multiplyModPrime(c[index3], aes_parallel->randNonZeroModPrime(t, nonZero_counter));
			}
		}
		else
		{
			//Single for loop
			a = 0;
			for (size_t k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				c[index3] = a;
				tempM = share_m[index3];

				bit_r = (smallType)((valueX >> (63-k)) & 1);

				if (bit_r == 0)
					a = addModPrime(a, tempM);
				else
					a = addModPrime(a, subtractModPrime((partyNum == PARTY_A), tempM));

				if (!beta[index2])
				{
					if (partyNum == PARTY_A)
						c[index3] = addModPrime(c[index3], 1+bit_r);
					c[index3] = subtractModPrime(c[index3], tempM);
				}
				else
				{
					if (partyNum == PARTY_A)
						c[index3] = addModPrime(c[index3], 1-bit_r);
					c[index3] = addModPrime(c[index3], tempM);
				}

				c[index3] = multiplyModPrime(c[index3], aes_parallel->randNonZeroModPrime(t, nonZero_counter));
			}
		}
		aes_parallel->AES_random_shuffle(c, index2*dim, (index2+1)*dim, t, shuffle_counter);
	}
	aes_parallel->counterIncrement();
}


// Private Compare functionality
void funcPrivateCompareMPC(const vector<smallType> &share_m, const vector<myType> &r, 
							const vector<smallType> &beta, vector<smallType> &betaPrime, 
							size_t size, size_t dim)
{
	log_print("funcPrivateCompareMPC");

	assert(dim == BIT_SIZE && "Private Compare assert issue");
	size_t sizeLong = size*dim;
	size_t index3, index2;
	size_t PARTY;

	if (THREE_PC)
		PARTY = PARTY_C;
	else if (FOUR_PC)
		PARTY = PARTY_D;


	if (PRIMARY)
	{
		smallType bit_r, a, tempM;
		vector<smallType> c(sizeLong);
		myType valueX;

		if (PARALLEL)
		{
			thread *threads = new thread[NO_CORES];
			int chunksize = size/NO_CORES;

			for (int i = 0; i < NO_CORES; i++)
			{
				int start = i*chunksize;
				int end = (i+1)*chunksize;
				if (i == NO_CORES - 1)
					end = size;
				
				threads[i] = thread(parallelPC, c.data(), start, end, i, share_m.data(), 
									r.data(), beta.data(), betaPrime.data(), dim);
				// threads[i] = thread(parallelPC, ref(c.data()), start, end, i, ref(share_m.data()), 
				// 					ref(r.data()), ref(beta.data()), ref(betaPrime.data()), dim);
			}

			for (int i = 0; i < NO_CORES; i++)
				threads[i].join();

			delete[] threads;
		}
		else
		{
			//Check the security of the first if condition
			for (size_t index2 = 0; index2 < size; ++index2)
			{
				if (beta[index2] == 1 and r[index2] != MINUS_ONE)
					valueX = r[index2] + 1;
				else
					valueX = r[index2];

				if (beta[index2] == 1 and r[index2] == MINUS_ONE)
				{
					//One share of zero and other shares of 1
					//Then multiply and shuffle
					for (size_t k = 0; k < dim; ++k)
					{
						index3 = index2*dim + k;
						c[index3] = aes_common->randModPrime();
						if (partyNum == PARTY_A)
							c[index3] = subtractModPrime((k!=0), c[index3]);

						c[index3] = multiplyModPrime(c[index3], aes_common->randNonZeroModPrime());
					}
				}
				else
				{
					//Single for loop
					a = 0;
					for (size_t k = 0; k < dim; ++k)
					{
						index3 = index2*dim + k;
						c[index3] = a;
						tempM = share_m[index3];

						bit_r = (smallType)((valueX >> (63-k)) & 1);

						if (bit_r == 0)
							a = addModPrime(a, tempM);
						else
							a = addModPrime(a, subtractModPrime((partyNum == PARTY_A), tempM));

						if (!beta[index2])
						{
							if (partyNum == PARTY_A)
								c[index3] = addModPrime(c[index3], 1+bit_r);
							c[index3] = subtractModPrime(c[index3], tempM);
						}
						else
						{
							if (partyNum == PARTY_A)
								c[index3] = addModPrime(c[index3], 1-bit_r);
							c[index3] = addModPrime(c[index3], tempM);
						}

						c[index3] = multiplyModPrime(c[index3], aes_common->randNonZeroModPrime());
					}
				}
				aes_common->AES_random_shuffle(c, index2*dim, (index2+1)*dim);
			}
		}
		sendVector<smallType>(c, PARTY, sizeLong);
	}

	if (partyNum == PARTY)
	{
		vector<smallType> c1(sizeLong);
		vector<smallType> c2(sizeLong);

		receiveVector<smallType>(c1, PARTY_A, sizeLong);
		receiveVector<smallType>(c2, PARTY_B, sizeLong);

		for (size_t index2 = 0; index2 < size; ++index2)
		{
			betaPrime[index2] = 0;
			for (int k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				if (addModPrime(c1[index3], c2[index3]) == 0)
				{
					betaPrime[index2] = 1;
					break;
				}	
			}
		}
	}
}

// Convert shares of a in \Z_L to shares in \Z_{L-1} (in place)
// a \neq L-1
void funcShareConvertMPC(vector<myType> &a, size_t size)
{
	log_print("funcShareConvertMPC");

	vector<myType> r(size);
	vector<smallType> etaDP(size);
	vector<smallType> alpha(size);
	vector<smallType> betai(size);
	vector<smallType> bit_shares(size*BIT_SIZE);
	vector<myType> delta_shares(size);
	vector<smallType> etaP(size);
	vector<myType> eta_shares(size);
	vector<myType> theta_shares(size);
	size_t PARTY;

	if (THREE_PC)
		PARTY = PARTY_C;
	else if (FOUR_PC)
		PARTY = PARTY_D;
	

	if (PRIMARY)
	{
		vector<myType> r1(size);
		vector<myType> r2(size);
		vector<myType> a_tilde(size);

		populateRandomVector<myType>(r1, size, "COMMON", "POSITIVE");
		populateRandomVector<myType>(r2, size, "COMMON", "POSITIVE");
		addVectors<myType>(r1, r2, r, size);

		if (partyNum == PARTY_A)
			wrapAround(r1, r2, alpha, size);

		if (partyNum == PARTY_A)
		{
			addVectors<myType>(a, r1, a_tilde, size);
			wrapAround(a, r1, betai, size);
		}
		if (partyNum == PARTY_B)
		{
			addVectors<myType>(a, r2, a_tilde, size);
			wrapAround(a, r2, betai, size);	
		}

		populateBitsVector(etaDP, "COMMON", size);
		sendVector<myType>(a_tilde, PARTY_C, size);
	}


	if (partyNum == PARTY_C)
	{
		vector<myType> x(size);
		vector<smallType> delta(size);
		vector<myType> a_tilde_1(size);	
		vector<myType> a_tilde_2(size);	
		vector<smallType> bit_shares_x_1(size*BIT_SIZE);
		vector<smallType> bit_shares_x_2(size*BIT_SIZE);
		vector<myType> delta_shares_1(size);
		vector<myType> delta_shares_2(size);

		receiveVector<myType>(a_tilde_1, PARTY_A, size);
		receiveVector<myType>(a_tilde_2, PARTY_B, size);
		addVectors<myType>(a_tilde_1, a_tilde_2, x, size);
		wrapAround(a_tilde_1, a_tilde_2, delta, size);
		sharesOfBits(bit_shares_x_1, bit_shares_x_2, x, size, "INDEP");

		sendVector<smallType>(bit_shares_x_1, PARTY_A, size*BIT_SIZE);
		sendVector<smallType>(bit_shares_x_2, PARTY_B, size*BIT_SIZE);
		sharesModuloOdd<smallType>(delta_shares_1, delta_shares_2, delta, size, "INDEP");
		sendVector<myType>(delta_shares_1, PARTY_A, size);
		sendVector<myType>(delta_shares_2, PARTY_B, size);	
	}

	if (PRIMARY)
	{
		receiveVector<smallType>(bit_shares, PARTY_C, size*BIT_SIZE);
		receiveVector<myType>(delta_shares, PARTY_C, size);
	}

	funcPrivateCompareMPC(bit_shares, r, etaDP, etaP, size, BIT_SIZE);

	if (partyNum == PARTY)
	{
		vector<myType> eta_shares_1(size);
		vector<myType> eta_shares_2(size);

		for (size_t i = 0; i < size; ++i)
			etaP[i] = 1 - etaP[i];

		sharesModuloOdd<smallType>(eta_shares_1, eta_shares_2, etaP, size, "INDEP");
		sendVector<myType>(eta_shares_1, PARTY_A, size);
		sendVector<myType>(eta_shares_2, PARTY_B, size);
	}

	if (PRIMARY)
	{
		receiveVector<myType>(eta_shares, PARTY, size);
		funcXORModuloOdd2PC(etaDP, eta_shares, theta_shares, size);
		addModuloOdd<myType, smallType>(theta_shares, betai, theta_shares, size);
		subtractModuloOdd<myType, myType>(theta_shares, delta_shares, theta_shares, size);

		if (partyNum == PARTY_A)
			subtractModuloOdd<myType, smallType>(theta_shares, alpha, theta_shares, size);

		subtractModuloOdd<myType, myType>(a, theta_shares, a, size);
	}
}


//Compute MSB of a and store it in b
//4PC: output is boolean shares of MSB in b
void funcComputeMSB4PC(const vector<myType> &a, vector<smallType> &b, size_t size)
{
	log_print("funcComputeMSB4PC");
	assert(FOUR_PC && "funcComputeMSB4PC called in non-4PC mode");

	vector<myType> ri(size);
	vector<smallType> bit_shares(size*BIT_SIZE);
	vector<smallType> LSB_shares(size);
	vector<smallType> beta(size);
	vector<myType> c(size);	
	vector<smallType> betaP(size);
	vector<smallType> gamma(size);
	vector<smallType> theta_shares(size);

	if (partyNum == PARTY_C)
	{
		vector<myType> r1(size);
		vector<myType> r2(size);
		vector<myType> r(size);
		vector<smallType> bit_shares_r_1(size*BIT_SIZE);
		vector<smallType> bit_shares_r_2(size*BIT_SIZE);
		vector<smallType> LSB_shares_1(size);
		vector<smallType> LSB_shares_2(size);

		for (size_t i = 0; i < size; ++i)
		{
			r1[i] = aes_indep->randModuloOdd();
			r2[i] = aes_indep->randModuloOdd();
		}

		addModuloOdd<myType, myType>(r1, r2, r, size);		
		sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "INDEP");
		sharesOfLSB(LSB_shares_1, LSB_shares_2, r, size, "INDEP");

		sendVector<myType>(r1, PARTY_A, size);
		sendVector<myType>(r2, PARTY_B, size);
		sendTwoVectors<smallType>(bit_shares_r_1, LSB_shares_1, PARTY_A, size*BIT_SIZE, size);
		sendTwoVectors<smallType>(bit_shares_r_2, LSB_shares_2, PARTY_B, size*BIT_SIZE, size);
	}

	if (PRIMARY)
	{
		vector<myType> temp(size);
		receiveVector<myType>(ri, PARTY_C, size);
		receiveTwoVectors<smallType>(bit_shares, LSB_shares, PARTY_C, size*BIT_SIZE, size);

		addModuloOdd<myType, myType>(a, a, c, size);
		addModuloOdd<myType, myType>(c, ri, c, size);

		thread *threads = new thread[2];

		threads[0] = thread(sendVector<myType>, ref(c), adversary(partyNum), size);
		threads[1] = thread(receiveVector<myType>, ref(temp), adversary(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		delete[] threads;

		//HEREEEEEEE
		// if (partyNum == PARTY_A)
		// 	sendVector<myType>(c, adversary(partyNum), size);
		// else
		// 	receiveVector<myType>(temp, adversary(partyNum), size);

		// if (partyNum == PARTY_B)
		// 	sendVector<myType>(c, adversary(partyNum), size);
		// else
		// 	receiveVector<myType>(temp, adversary(partyNum), size);

		// sendVector<myType>(c, adversary(partyNum), size);
		// receiveVector<myType>(temp, adversary(partyNum), size);

		addModuloOdd<myType, myType>(c, temp, c, size);
		populateBitsVector(beta, "COMMON", size);
	}

	funcPrivateCompareMPC(bit_shares, c, beta, betaP, size, BIT_SIZE);

	if (partyNum == PARTY_D)
	{
		vector<smallType> theta_shares_1(size);
		vector<smallType> theta_shares_2(size);

		sharesOfBitVector(theta_shares_1, theta_shares_2, betaP, size, "INDEP");
		sendVector<smallType>(theta_shares_1, PARTY_A, size);
		sendVector<smallType>(theta_shares_2, PARTY_B, size);
	}

	if (PRIMARY)
	{
		// theta_shares is the same as gamma (in older versions);
		// LSB_shares is the same as delta (in older versions);
		receiveVector<smallType>(theta_shares, PARTY_D, size);
		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				theta_shares[i] = theta_shares[i] ^ beta[i];

		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				if (c[i] & 1)
					LSB_shares[i] = LSB_shares[i] ^ 1;
		
		for (size_t i = 0; i < size; ++i)
			b[i] = theta_shares[i] ^ LSB_shares[i];
	}
}


//Compute MSB of a and store it in b
//3PC: output is shares of MSB in \Z_L
void funcComputeMSB3PC(const vector<myType> &a, vector<myType> &b, size_t size)
{
	log_print("funcComputeMSB3PC");
	assert(THREE_PC && "funcComputeMSB3PC called in non-3PC mode");
	
	vector<myType> ri(size);
	vector<smallType> bit_shares(size*BIT_SIZE);
	vector<myType> LSB_shares(size);
	vector<smallType> beta(size);
	vector<myType> c(size);	
	vector<smallType> betaP(size);
	vector<smallType> gamma(size);
	vector<myType> theta_shares(size);

	if (partyNum == PARTY_C)
	{
		vector<myType> r1(size);
		vector<myType> r2(size);
		vector<myType> r(size);
		vector<smallType> bit_shares_r_1(size*BIT_SIZE);
		vector<smallType> bit_shares_r_2(size*BIT_SIZE);
		vector<myType> LSB_shares_1(size);
		vector<myType> LSB_shares_2(size);

		for (size_t i = 0; i < size; ++i)
		{
			r1[i] = aes_indep->randModuloOdd();
			r2[i] = aes_indep->randModuloOdd();
		}

		addModuloOdd<myType, myType>(r1, r2, r, size);		
		sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "INDEP");
		sharesOfLSB(LSB_shares_1, LSB_shares_2, r, size, "INDEP");

		sendVector<smallType>(bit_shares_r_1, PARTY_A, size*BIT_SIZE);
		sendVector<smallType>(bit_shares_r_2, PARTY_B, size*BIT_SIZE);
		sendTwoVectors<myType>(r1, LSB_shares_1, PARTY_A, size, size);
		sendTwoVectors<myType>(r2, LSB_shares_2, PARTY_B, size, size);
	}

	if (PRIMARY)
	{
		vector<myType> temp(size);
		receiveVector<smallType>(bit_shares, PARTY_C, size*BIT_SIZE);
		receiveTwoVectors<myType>(ri, LSB_shares, PARTY_C, size, size);

		addModuloOdd<myType, myType>(a, a, c, size);
		addModuloOdd<myType, myType>(c, ri, c, size);

		thread *threads = new thread[2];

		threads[0] = thread(sendVector<myType>, ref(c), adversary(partyNum), size);
		threads[1] = thread(receiveVector<myType>, ref(temp), adversary(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		delete[] threads;

		//HEREEEEEEE
		// if (partyNum == PARTY_A)
		// 	sendVector<myType>(c, adversary(partyNum), size);
		// else
		// 	receiveVector<myType>(temp, adversary(partyNum), size);

		// if (partyNum == PARTY_B)
		// 	sendVector<myType>(c, adversary(partyNum), size);
		// else
		// 	receiveVector<myType>(temp, adversary(partyNum), size);		


		// sendVector<myType>(c, adversary(partyNum), size);
		// receiveVector<myType>(temp, adversary(partyNum), size);

		addModuloOdd<myType, myType>(c, temp, c, size);
		populateBitsVector(beta, "COMMON", size);
	}

	funcPrivateCompareMPC(bit_shares, c, beta, betaP, size, BIT_SIZE);

	if (partyNum == PARTY_C)
	{
		vector<myType> theta_shares_1(size);
		vector<myType> theta_shares_2(size);

		sharesOfBitVector(theta_shares_1, theta_shares_2, betaP, size, "INDEP");
		sendVector<myType>(theta_shares_1, PARTY_A, size);
		sendVector<myType>(theta_shares_2, PARTY_B, size);
	}

	vector<myType> prod(size), temp(size);
	if (PRIMARY)
	{
		// theta_shares is the same as gamma (in older versions);
		// LSB_shares is the same as delta (in older versions);
		receiveVector<myType>(theta_shares, PARTY_C, size);
		
		myType j = 0;
		if (partyNum == PARTY_A)
			j = floatToMyType(1);

		for (size_t i = 0; i < size; ++i)
			theta_shares[i] = (1 - 2*beta[i])*theta_shares[i] + j*beta[i];

		for (size_t i = 0; i < size; ++i)
			LSB_shares[i] = (1 - 2*(c[i] & 1))*LSB_shares[i] + j*(c[i] & 1);		
	}

	funcDotProductMPC(theta_shares, LSB_shares, prod, size);

	if (PRIMARY)
	{
		populateRandomVector<myType>(temp, size, "COMMON", "NEGATIVE");
		for (size_t i = 0; i < size; ++i)
			b[i] = theta_shares[i] + LSB_shares[i] - 2*prod[i] + temp[i];
	}
}


// 4PC: All parties start with shares of something (natural or unnatural) in a 
// and want selected shares with PARTY_A and PARTY_B (using b) in c. 
void funcSelectShares4PC(const vector<myType> &a, const vector<smallType> &b, vector<myType> &c, size_t size)
{
	log_print("funcSelectShares4PC");

	vector<smallType> beta(size);
	vector<smallType> gamma(size);
	vector<myType> u(size);
	vector<myType> v(size);

	if (PRIMARY)
	{
		populateBitsVector(beta, "COMMON", size);

		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				gamma[i] = b[i];

		if (partyNum == PARTY_B)
			XORVectors(beta, b, gamma, size);

		sendVector<smallType>(gamma, PARTY_C, size);
		sendVector<smallType>(gamma, PARTY_D, size);
	}

	vector<myType> a_temp = a;
	getVectorfromPrimary<myType>(a_temp, size, "RANDOMIZE", "NATURAL");

	if (NON_PRIMARY)
	{
		vector<smallType> temp(size);
		vector<myType> tempZeros(size), minusA(size);

		receiveVector<smallType>(gamma, PARTY_A, size);
		receiveVector<smallType>(temp, PARTY_B, size);
		XORVectors(gamma, temp, gamma, size);

		populateRandomVector<myType>(tempZeros, size, "COMMON", "NEGATIVE");
		subtractVectors<myType>(tempZeros, a_temp, minusA, size);
		populateRandomVector<myType>(tempZeros, size, "COMMON", "NEGATIVE");

		funcConditionalSet2PC(minusA, tempZeros, gamma, u, v, size);	
		sendTwoVectors<myType>(u, v, partner(partyNum), size, size);
	}

	if (PRIMARY)
	{
		receiveTwoVectors<myType>(u, v, partner(partyNum), size, size);

		for (size_t i = 0; i < size; ++i)
		{
			if (beta[i] == 0)
				c[i] = a[i] + u[i];
			else
				c[i] = a[i] + v[i];
		}
	}
}

// 3PC SelectShares: c contains shares of selector bit (encoded in myType). 
// a,b,c are shared across PARTY_A, PARTY_B
void funcSelectShares3PC(const vector<myType> &a, const vector<myType> &b, 
								vector<myType> &c, size_t size)
{
	log_print("funcSelectShares3PC");

	assert(THREE_PC && "funcSelectShares3PC called in non-3PC mdoe");
	funcDotProductMPC(a, b, c, size);
}

// 4PC: PARTY_A, PARTY_B hold shares in a, want shares of RELU' in b.
void funcRELUPrime4PC(const vector<myType> &a, vector<smallType> &b, size_t size)
{
	log_print("funcRELUPrime4PC");
	assert(FOUR_PC && "funcRELUPrime4PC called in non-4PC mode");

	vector<myType> twoA(size);

	for (size_t i = 0; i < size; ++i)
		twoA[i] = (a[i] << 1);

	funcShareConvertMPC(twoA, size);
	funcComputeMSB4PC(twoA, b, size);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			b[i] = 1 - b[i];
}

// 3PC: PARTY_A, PARTY_B hold shares in a, want shares of RELU' in b.
void funcRELUPrime3PC(const vector<myType> &a, vector<myType> &b, size_t size)
{
	log_print("funcRELUPrime3PC");
	assert(THREE_PC && "funcRELUPrime3PC called in non-3PC mode");

	vector<myType> twoA(size, 0);
	myType j = 0;

	for (size_t i = 0; i < size; ++i)
		twoA[i] = (a[i] << 1);

	funcShareConvertMPC(twoA, size);
	funcComputeMSB3PC(twoA, b, size);

	if (partyNum == PARTY_A)
		j = floatToMyType(1);

	if (PRIMARY)
		for (size_t i = 0; i < size; ++i)
			b[i] = j - b[i];
}

//PARTY_A, PARTY_B hold shares in a, want shares of RELU in b.
void funcRELUMPC(const vector<myType> &a, vector<myType> &b, size_t size)
{
	log_print("funcRELUMPC");

	if (FOUR_PC)
	{
		vector<smallType> reluPrime(size);

		funcRELUPrime4PC(a, reluPrime, size);
		funcSelectShares4PC(a, reluPrime, b, size);
	}

	if (THREE_PC)
	{
		vector<myType> reluPrime(size);

		funcRELUPrime3PC(a, reluPrime, size);
		funcSelectShares3PC(a, reluPrime, b, size);
	}
}


//All parties start with shares of a number in a and b and the quotient is in quotient.
void funcDivisionMPC(const vector<myType> &a, const vector<myType> &b, vector<myType> &quotient, 
							size_t size)
{
	log_print("funcDivisionMPC");

	if (THREE_PC)
	{
		vector<myType> varQ(size, 0); 
		vector<myType> varP(size, 0); 
		vector<myType> varD(size, 0); 
		vector<myType> tempZeros(size, 0);
		vector<myType> varB(size, 0);
		vector<myType> input_1(size, 0), input_2(size, 0); 

		for (size_t i = 0; i < size; ++i)
		{
			varP[i] = 0;
			quotient[i] = 0;
		}

		for (size_t looper = 1; looper < FLOAT_PRECISION+1; ++looper)
		{
			if (PRIMARY)
			{
				for (size_t i = 0; i < size; ++i)
					input_1[i] = -b[i];

				funcTruncate2PC(input_1, looper, size, PARTY_A, PARTY_B);
				addVectors<myType>(input_1, a, input_1, size);
				subtractVectors<myType>(input_1, varP, input_1, size);
			}
			funcRELUPrime3PC(input_1, varB, size);

			//Get the required shares of y/2^i and 2^FLOAT_PRECISION/2^i in input_1 and input_2
			for (size_t i = 0; i < size; ++i)
					input_1[i] = b[i];

			if (PRIMARY)
				funcTruncate2PC(input_1, looper, size, PARTY_A, PARTY_B);

			if (partyNum == PARTY_A)
				for (size_t i = 0; i < size; ++i)
					input_2[i] = (1 << FLOAT_PRECISION);

			if (partyNum == PARTY_B)
				for (size_t i = 0; i < size; ++i)
					input_2[i] = 0;

			if (PRIMARY)
				funcTruncate2PC(input_2, looper, size, PARTY_A, PARTY_B);

			// funcSelectShares3PC(input_1, varB, varD, size);
			// funcSelectShares3PC(input_2, varB, varQ, size);

			vector<myType> A_one(size, 0), B_one(size, 0), C_one(size, 0);
			vector<myType> A_two(size, 0), B_two(size, 0), C_two(size, 0);

			if (HELPER)
			{
				vector<myType> A1_one(size, 0), A2_one(size, 0), 
							   B1_one(size, 0), B2_one(size, 0), 
							   C1_one(size, 0), C2_one(size, 0);

				vector<myType> A1_two(size, 0), A2_two(size, 0), 
							   B1_two(size, 0), B2_two(size, 0), 
							   C1_two(size, 0), C2_two(size, 0);

				populateRandomVector<myType>(A1_one, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(A2_one, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(B1_one, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(B2_one, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(A1_two, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(A2_two, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(B1_two, size, "INDEP", "POSITIVE");
				populateRandomVector<myType>(B2_two, size, "INDEP", "POSITIVE");


				addVectors<myType>(A1_one, A2_one, A_one, size);
				addVectors<myType>(B1_one, B2_one, B_one, size);
				addVectors<myType>(A1_two, A2_two, A_two, size);
				addVectors<myType>(B1_two, B2_two, B_two, size);

				for (size_t i = 0; i < size; ++i)
					C_one[i] = A_one[i] * B_one[i];

				for (size_t i = 0; i < size; ++i)
					C_two[i] = A_two[i] * B_two[i];

				splitIntoShares(C_one, C1_one, C2_one, size);
				splitIntoShares(C_two, C1_two, C2_two, size);

				sendSixVectors<myType>(A1_one, B1_one, C1_one, A1_two, B1_two, C1_two, PARTY_A, size, size, size, size, size, size);
				sendSixVectors<myType>(A2_one, B2_one, C2_one, A2_two, B2_two, C2_two, PARTY_B, size, size, size, size, size, size);
				// sendThreeVectors<myType>(A1_one, B1_one, C1_one, PARTY_A, size, size, size);
				// sendThreeVectors<myType>(A2_one, B2_one, C2_one, PARTY_B, size, size, size);
				// sendThreeVectors<myType>(A1_two, B1_two, C1_two, PARTY_A, size, size, size);
				// sendThreeVectors<myType>(A2_two, B2_two, C2_two, PARTY_B, size, size, size);

			}

			if (PRIMARY)
			{
				receiveSixVectors<myType>(A_one, B_one, C_one, A_two, B_two, C_two, PARTY_C, size, size, size, size, size, size);
				// receiveThreeVectors<myType>(A_one, B_one, C_one, PARTY_C, size, size, size);
				// receiveThreeVectors<myType>(A_two, B_two, C_two, PARTY_C, size, size, size);
				
				vector<myType> E_one(size), F_one(size), temp_E_one(size), temp_F_one(size);
				vector<myType> E_two(size), F_two(size), temp_E_two(size), temp_F_two(size);
				myType temp_one, temp_two;

				subtractVectors<myType>(input_1, A_one, E_one, size);
				subtractVectors<myType>(varB, B_one, F_one, size);
				subtractVectors<myType>(input_2, A_two, E_two, size);
				subtractVectors<myType>(varB, B_two, F_two, size);


				thread *threads = new thread[2];

				threads[0] = thread(sendFourVectors<myType>, ref(E_one), ref(F_one), ref(E_two), ref(F_two), adversary(partyNum), size, size, size, size);
				threads[1] = thread(receiveFourVectors<myType>, ref(temp_E_one), ref(temp_F_one), ref(temp_E_two), ref(temp_F_two), adversary(partyNum), size, size, size, size);

				for (int i = 0; i < 2; i++)
					threads[i].join();

				delete[] threads;

				//HEREEEEEEE
				// if (partyNum == PARTY_A)
				// 	sendFourVectors<myType>(E_one, F_one, E_two, F_two, adversary(partyNum), size, size, size, size);
				// else
				// 	receiveFourVectors<myType>(temp_E_one, temp_F_one, temp_E_two, temp_F_two, adversary(partyNum), size, size, size, size);	

				// if (partyNum == PARTY_B)
				// 	sendFourVectors<myType>(E_one, F_one, E_two, F_two, adversary(partyNum), size, size, size, size);
				// else
				// 	receiveFourVectors<myType>(temp_E_one, temp_F_one, temp_E_two, temp_F_two, adversary(partyNum), size, size, size, size);	


				// sendTwoVectors<myType>(E_one, F_one, adversary(partyNum), size, size);
				// receiveTwoVectors<myType>(temp_E_one, temp_F_one, adversary(partyNum), size, size);
				// sendTwoVectors<myType>(E_two, F_two, adversary(partyNum), size, size);
				// receiveTwoVectors<myType>(temp_E_two, temp_F_two, adversary(partyNum), size, size);


				addVectors<myType>(E_one, temp_E_one, E_one, size);
				addVectors<myType>(F_one, temp_F_one, F_one, size);
				addVectors<myType>(E_two, temp_E_two, E_two, size);
				addVectors<myType>(F_two, temp_F_two, F_two, size);

				for (size_t i = 0; i < size; ++i)
				{
					varD[i] = input_1[i] * F_one[i];
					temp_one = E_one[i] * varB[i];
					varD[i] = varD[i] + temp_one;

					if (partyNum == PARTY_A)
					{
						temp_one = E_one[i] * F_one[i];
						varD[i] = varD[i] - temp_one;
					}
				}
				
				for (size_t i = 0; i < size; ++i)
				{
					varQ[i] = input_2[i] * F_two[i];
					temp_two = E_two[i] * varB[i];
					varQ[i] = varQ[i] + temp_two;

					if (partyNum == PARTY_A)
					{
						temp_two = E_two[i] * F_two[i];
						varQ[i] = varQ[i] - temp_two;
					}
				}

				addVectors<myType>(varD, C_one, varD, size);
				funcTruncate2PC(varD, FLOAT_PRECISION, size, PARTY_A, PARTY_B);

				addVectors<myType>(varQ, C_two, varQ, size);
				funcTruncate2PC(varQ, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
			}

			addVectors<myType>(varP, varD, varP, size);
			addVectors<myType>(quotient, varQ, quotient, size);
		}
	}

	if (FOUR_PC)
	{
		vector<myType> varQ(size); 
		vector<myType> varP(size); 
		vector<myType> varD(size); 
		vector<myType> tempZeros(size);
		vector<smallType> varB(size);
		vector<myType> input_1(size), input_2(size); 

		//To split open SelectShare protocol.
		vector<smallType> beta_1(size), beta_2(size);
		vector<smallType> gamma_1(size), gamma_2(size);
		vector<myType> u_1(size), u_2(size);
		vector<myType> v_1(size), v_2(size);

		for (size_t i = 0; i < size; ++i)
		{
			varP[i] = 0;
			quotient[i] = 0;
		}


		for (size_t looper = 1; looper < FLOAT_PRECISION+1; ++looper)
		{
			if (PRIMARY)
			{
				for (size_t i = 0; i < size; ++i)
					input_1[i] = -b[i];

				funcTruncate2PC(input_1, looper, size, PARTY_A, PARTY_B);
				addVectors<myType>(input_1, a, input_1, size);
				subtractVectors<myType>(input_1, varP, input_1, size);
			}
			funcRELUPrime4PC(input_1, varB, size);
			

			//Get the required shares of y/2^i and 2^FLOAT_PRECISION/2^i in input_1 and input_2
			for (size_t i = 0; i < size; ++i)
					input_1[i] = b[i];

			if (PRIMARY)
				funcTruncate2PC(input_1, looper, size, PARTY_A, PARTY_B);
			if (NON_PRIMARY)
				funcTruncate2PC(input_1, looper, size, PARTY_C, PARTY_D);

			populateRandomVector<myType>(tempZeros, size, "COMMON", "NEGATIVE");
			addVectors<myType>(input_1, tempZeros, input_1, size);

			if (partyNum == PARTY_A or partyNum == PARTY_C)
				for (size_t i = 0; i < size; ++i)
					input_2[i] = 1 << FLOAT_PRECISION;

			if (partyNum == PARTY_B or partyNum == PARTY_D)
				for (size_t i = 0; i < size; ++i)
					input_2[i] = 0;

			if (PRIMARY)
				funcTruncate2PC(input_2, looper, size, PARTY_A, PARTY_B);
			if (NON_PRIMARY)
				funcTruncate2PC(input_2, looper, size, PARTY_C, PARTY_D);

			populateRandomVector<myType>(tempZeros, size, "COMMON", "NEGATIVE");
			addVectors<myType>(input_2, tempZeros, input_2, size);


			//Split open the SelectShares protocol
			if (PRIMARY)
			{
				populateBitsVector(beta_1, "COMMON", size);
				if (partyNum == PARTY_A)
					for (size_t i = 0; i < size; ++i)
						gamma_1[i] = varB[i];

				if (partyNum == PARTY_B)
					XORVectors(beta_1, varB, gamma_1, size);

				populateBitsVector(beta_2, "COMMON", size);
				if (partyNum == PARTY_A)
					for (size_t i = 0; i < size; ++i)
						gamma_2[i] = varB[i];

				if (partyNum == PARTY_B)
					XORVectors(beta_2, varB, gamma_2, size);

				sendTwoVectors<smallType>(gamma_1, gamma_2, PARTY_C, size, size);
				sendTwoVectors<smallType>(gamma_1, gamma_2, PARTY_D, size, size);
			}

			if (NON_PRIMARY)
			{
				vector<smallType> temp_1(size), temp_2(size);
				vector<myType> tempZeros_1(size), tempZeros_2(size), minusInput_1(size), minusInput_2(size);

				receiveTwoVectors<smallType>(gamma_1, gamma_2, PARTY_A, size, size);
				receiveTwoVectors<smallType>(temp_1, temp_2, PARTY_B, size, size);

				XORVectors(gamma_1, temp_1, gamma_1, size);
				XORVectors(gamma_2, temp_2, gamma_2, size);

				populateRandomVector<myType>(tempZeros_1, size, "COMMON", "NEGATIVE");
				subtractVectors<myType>(tempZeros_1, input_1, minusInput_1, size);
				populateRandomVector<myType>(tempZeros_1, size, "COMMON", "NEGATIVE");
				funcConditionalSet2PC(minusInput_1, tempZeros_1, gamma_1, u_1, v_1, size);	

				populateRandomVector<myType>(tempZeros_2, size, "COMMON", "NEGATIVE");
				subtractVectors<myType>(tempZeros_2, input_2, minusInput_2, size);
				populateRandomVector<myType>(tempZeros_2, size, "COMMON", "NEGATIVE");
				funcConditionalSet2PC(minusInput_2, tempZeros_2, gamma_2, u_2, v_2, size);

				sendFourVectors<myType>(u_1, v_1, u_2, v_2, partner(partyNum), size, size, size, size);
			}

			if (PRIMARY)
			{
				receiveFourVectors<myType>(u_1, v_1, u_2, v_2, partner(partyNum), size, size, size, size);

				for (size_t i = 0; i < size; ++i)
				{
					if (beta_1[i] == 0)
						varD[i] = input_1[i] + u_1[i];
					else
						varD[i] = input_1[i] + v_1[i];
				}

				for (size_t i = 0; i < size; ++i)
				{
					if (beta_2[i] == 0)
						varQ[i] = input_2[i] + u_2[i];
					else
						varQ[i] = input_2[i] + v_2[i];
				}	
			}

			addVectors<myType>(varP, varD, varP, size);
			addVectors<myType>(quotient, varQ, quotient, size);
		}		
	}
}



//Chunk wise maximum of a vector of size rows*columns and maximum is caclulated of every 
//column number of elements. max is a vector of size rows. maxIndex contains the index of 
//the maximum value.
//PARTY_A, PARTY_B start with the shares in a and {A,B} and {C,D} have the results in 
//max and maxIndex.
void funcMaxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	log_print("funcMaxMPC");

	if (THREE_PC)
	{
		vector<myType> diff(rows), diffIndex(rows), rp(rows), indexShares(rows*columns, 0);

		for (size_t i = 0; i < rows; ++i)
		{
			max[i] = a[i*columns];
			maxIndex[i] = 0;
		}

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				if (partyNum == PARTY_A)
					indexShares[i*columns + j] = j;

		for (size_t i = 1; i < columns; ++i)
		{
			for (size_t	j = 0; j < rows; ++j)
				diff[j] = max[j] - a[j*columns + i];

			for (size_t	j = 0; j < rows; ++j)
				diffIndex[j] = maxIndex[j] - indexShares[j*columns + i];

			funcRELUPrime3PC(diff, rp, rows);
			funcSelectShares3PC(diff, rp, max, rows);
			funcSelectShares3PC(diffIndex, rp, maxIndex, rows);

			for (size_t	j = 0; j < rows; ++j)
				max[j] = max[j] + a[j*columns + i];

			for (size_t	j = 0; j < rows; ++j)
				maxIndex[j] = maxIndex[j] + indexShares[j*columns + i];
		}
	}

	if (FOUR_PC)
	{
		vector<myType> diff(rows), diffIndex(rows), indexShares(rows*columns, 0);
		vector<smallType> rp(rows);

		// getVectorfromPrimary<myType>(a, rows*columns, "RANDOMIZE", "NATURAL");

		for (size_t i = 0; i < rows; ++i)
		{
			max[i] = a[i*columns];
			maxIndex[i] = 0;
		}

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				if (partyNum == PARTY_A or partyNum == PARTY_C)
					indexShares[i*columns + j] = j;
				if (partyNum == PARTY_B or partyNum == PARTY_D)
					indexShares[i*columns + j] = 0;				
			}

		for (size_t i = 1; i < columns; ++i)
		{
			for (size_t	j = 0; j < rows; ++j)
				diff[j] = max[j] - a[j*columns + i];

			for (size_t	j = 0; j < rows; ++j)
				diffIndex[j] = maxIndex[j] - indexShares[j*columns + i];

			funcRELUPrime4PC(diff, rp, rows);
			funcSelectShares4PC(diff, rp, max, rows);
			funcSelectShares4PC(diffIndex, rp, maxIndex, rows);

			for (size_t	j = 0; j < rows; ++j)
				max[j] = max[j] + a[j*columns + i];

			for (size_t	j = 0; j < rows; ++j)
				maxIndex[j] = maxIndex[j] + indexShares[j*columns + i];

			// getVectorfromPrimary<myType>(max, rows, "AS-IS", "NATURAL");
			// getVectorfromPrimary<myType>(maxIndex, rows, "AS-IS", "NATURAL");
		}
	}
}


//MaxIndex is of size rows. a is of size rows*columns.
//a will be set to 0's except at maxIndex (in every set of column)
void funcMaxIndexMPC(vector<myType> &a, const vector<myType> &maxIndex, 
						size_t rows, size_t columns)
{
	log_print("funcMaxIndexMPC");
	assert(((1 << (BIT_SIZE-1)) % columns) == 0 && "funcMaxIndexMPC works only for power of 2 columns");
	assert(columns < 257 && "This implementation does not support larger than 257 columns");
	
	vector<smallType> random(rows);

	if (PRIMARY)
	{
		vector<smallType> toSend(rows);
		for (size_t i = 0; i < rows; ++i)
			toSend[i] = (smallType)maxIndex[i] % columns;
		
		populateRandomVector<smallType>(random, rows, "COMMON", "POSITIVE");
		if (partyNum == PARTY_A)
			addVectors<smallType>(toSend, random, toSend, rows);

		sendVector<smallType>(toSend, PARTY_C, rows);
	}

	if (partyNum == PARTY_C)
	{
		vector<smallType> index(rows), temp(rows);
		vector<myType> vector(rows*columns, 0), share_1(rows*columns), share_2(rows*columns);
		receiveVector<smallType>(index, PARTY_A, rows);
		receiveVector<smallType>(temp, PARTY_B, rows);
		addVectors<smallType>(index, temp, index, rows);

		for (size_t i = 0; i < rows; ++i)
			index[i] = index[i] % columns;

		for (size_t i = 0; i < rows; ++i)
			vector[i*columns + index[i]] = 1;

		splitIntoShares(vector, share_1, share_2, rows*columns);
		sendVector<myType>(share_1, PARTY_A, rows*columns);
		sendVector<myType>(share_2, PARTY_B, rows*columns);
	}

	if (PRIMARY)
	{
		receiveVector<myType>(a, PARTY_C, rows*columns);
		size_t offset = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			rotate(a.begin()+offset, a.begin()+offset+(random[i] % columns), a.begin()+offset+columns);
			offset += columns;
		}
	}
}


extern CommunicationObject commObject;
void aggregateCommunication()
{
	vector<myType> vec(4, 0), temp(4, 0);
	vec[0] = commObject.getSent();
	vec[1] = commObject.getRecv();
	vec[2] = commObject.getRoundsSent();
	vec[3] = commObject.getRoundsRecv();

	if (THREE_PC)
	{
		if (partyNum == PARTY_B or partyNum == PARTY_C)
			sendVector<myType>(vec, PARTY_A, 4);

		if (partyNum == PARTY_A)
		{
			receiveVector<myType>(temp, PARTY_B, 4);
			addVectors<myType>(vec, temp, vec, 4);
			receiveVector<myType>(temp, PARTY_C, 4);
			addVectors<myType>(vec, temp, vec, 4);
		}
	}

	if (FOUR_PC)
	{
		if (partyNum == PARTY_B or partyNum == PARTY_C or partyNum == PARTY_D)
			sendVector<myType>(vec, PARTY_A, 4);

		if (partyNum == PARTY_A)
		{
			receiveVector<myType>(temp, PARTY_B, 4);
			addVectors<myType>(vec, temp, vec, 4);
			receiveVector<myType>(temp, PARTY_C, 4);
			addVectors<myType>(vec, temp, vec, 4);
			receiveVector<myType>(temp, PARTY_D, 4);
			addVectors<myType>(vec, temp, vec, 4);
		}
	}

	if (partyNum == PARTY_A)
	{
		cout << "------------------------------------" << endl;
		cout << "Total communication: " << (float)vec[0]/1000000 << "MB (sent) and " << (float)vec[1]/1000000 << "MB (recv)\n";
		cout << "Total calls: " << vec[2] << " (sends) and " << vec[3] << " (recvs)" << endl;
		cout << "------------------------------------" << endl;
	}
}


/******************************** Debug ********************************/
void debugDotProd()
{
	size_t size = 10;
	vector<myType> a(size, 0), b(size, 0), c(size);
	vector<myType> temp(size);

	populateRandomVector<myType>(temp, size, "COMMON", "NEGATIVE");
	for (size_t i = 0; i < size; ++i)
	{
		if (partyNum == PARTY_A)
			a[i] = temp[i] + floatToMyType(i);
		else
			a[i] = temp[i];
	}

	populateRandomVector<myType>(temp, size, "COMMON", "NEGATIVE");
	for (size_t i = 0; i < size; ++i)
	{
		if (partyNum == PARTY_A)
			b[i] = temp[i] + floatToMyType(i);
		else
			b[i] = temp[i];
	}

	// if (PRIMARY)
	// 	for (size_t i = 0; i < size; ++i)
	// 		a[i] = aes_indep->get64Bits();


	// if (PRIMARY)
	// 	for (size_t i = 0; i < size; ++i)
	// 		b[i] = aes_indep->get64Bits();

	funcDotProductMPC(a, b, c, size);

	if (PRIMARY)
		funcReconstruct2PC(c, size, "c");
}

void debugComputeMSB()
{
	size_t size = 10;
	vector<myType> a(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			a[i] = i - 5;

	if (THREE_PC)
	{
		vector<myType> c(size);
		funcComputeMSB3PC(a, c, size);

		if (PRIMARY)
			funcReconstruct2PC(c, size, "c");
	}

	if (FOUR_PC)
	{
		vector<smallType> c(size);
		funcComputeMSB4PC(a, c, size);
		
		if (PRIMARY)
			funcReconstructBit2PC(c, size, "c");
	}	
}

void debugPC()
{
	size_t size = 10;
	vector<myType> r(size);
	vector<smallType> bit_shares(size*BIT_SIZE, 0);
	
	for (size_t i = 0; i < size; ++i)
		r[i] = 5+i;

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			for (size_t j = 0; j < BIT_SIZE; ++j)
				if (j == BIT_SIZE - 1 - i)
					bit_shares[i*BIT_SIZE + j] = 1;

	vector<smallType> beta(size);
	vector<smallType> betaPrime(size);

	if (PRIMARY)
		populateBitsVector(beta, "COMMON", size);

	funcPrivateCompareMPC(bit_shares, r, beta, betaPrime, size, BIT_SIZE);

	if (PRIMARY)
		for (size_t i = 0; i < size; ++i)
			cout << (int) beta[i] << endl; 

	if (partyNum == PARTY_D)
	{
		for (size_t i = 0; i < size; ++i)
			cout << (int)(1 << i) << " " << (int) r[i] << " "
				 << (int) betaPrime[i] << endl; 
	}
}

void debugDivision()
{
	size_t size = 10;
	vector<myType> numerator(size);
	vector<myType> denominator(size);
	vector<myType> quotient(size,0);
	
	for (size_t i = 0; i < size; ++i)
		numerator[i] = 50;

	for (size_t i = 0; i < size; ++i)
		denominator[i] = 50*size;

	funcDivisionMPC(numerator, denominator, quotient, size);

	if (PRIMARY)
	{
		funcReconstruct2PC(numerator, size, "Numerator");
		funcReconstruct2PC(denominator, size, "Denominator");
		funcReconstruct2PC(quotient, size, "Quotient");
	}
}

void debugMax()
{
	size_t rows = 1;
	size_t columns = 10;
	vector<myType> a(rows*columns, 0);

	if (partyNum == PARTY_A or partyNum == PARTY_C){
		a[0] = 0; a[1] = 1; a[2] = 0; a[3] = 4; a[4] = 5; 
		a[5] = 3; a[6] = 10; a[7] = 6, a[8] = 41; a[9] = 9;
	}

	vector<myType> max(rows), maxIndex(rows);
	funcMaxMPC(a, max, maxIndex, rows, columns);

	if (PRIMARY)
	{
		funcReconstruct2PC(a, columns, "a");
		funcReconstruct2PC(max, rows, "max");
		funcReconstruct2PC(maxIndex, rows, "maxIndex");
		cout << "-----------------" << endl;
	}
}


void debugSS()
{
	size_t size = 10;
	vector<myType> inputs(size, 0), outputs(size, 0);

	if (FOUR_PC)
	{
		vector<smallType> selector(size, 0);
	
		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				selector[i] = aes_indep->getBit();

		if (PRIMARY)
			funcReconstructBit2PC(selector, size, "selector");

		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				inputs[i] = aes_indep->get8Bits();

		getVectorfromPrimary(inputs, size, "RANDOMIZE", "NATURAL");

		funcSelectShares4PC(inputs, selector, outputs, size);

		if (PRIMARY)
		{
			funcReconstruct2PC(inputs, size, "inputs");
			funcReconstruct2PC(outputs, size, "outputs");
		}
	}

	if (THREE_PC)
	{
		vector<myType> selector(size, 0);

		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				selector[i] = (myType)(aes_indep->getBit() << FLOAT_PRECISION);

		if (PRIMARY)
			funcReconstruct2PC(selector, size, "selector");

		if (partyNum == PARTY_A)
			for (size_t i = 0; i < size; ++i)
				inputs[i] = (myType)aes_indep->get8Bits();

		funcSelectShares3PC(inputs, selector, outputs, size);

		if (PRIMARY)
		{
			funcReconstruct2PC(inputs, size, "inputs");
			funcReconstruct2PC(outputs, size, "outputs");
		}
	}
}


void debugMatMul()
{
	size_t rows = 3; 
	size_t common_dim = 2;
	size_t columns = 3;
	// size_t rows = 784; 
	// size_t common_dim = 128;
	// size_t columns = 128;
 	size_t transpose_a = 0, transpose_b = 0;

	vector<myType> a(rows*common_dim);
	vector<myType> b(common_dim*columns);
	vector<myType> c(rows*columns);

	for (size_t i = 0; i < a.size(); ++i)
		a[i] = floatToMyType(i);

	for (size_t i = 0; i < b.size(); ++i)
		b[i] = floatToMyType(i);

	if (PRIMARY)
		funcReconstruct2PC(a, a.size(), "a");

	if (PRIMARY)
		funcReconstruct2PC(b, b.size(), "b");

	funcMatMulMPC(a, b, c, rows, common_dim, columns, transpose_a, transpose_b);
	// for (int i = 0; i < 10; ++i){
	// 	funcMatMulMPC(a, b, c, rows, common_dim, columns, transpose_a, transpose_b);
	// 	c[0]++;
	// }

	if (PRIMARY)
		funcReconstruct2PC(c, c.size(), "c");
}

void debugReLUPrime()
{
	size_t size = 10;
	vector<myType> inputs(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			inputs[i] = aes_indep->get8Bits() - aes_indep->get8Bits();

	if (THREE_PC)
	{
		vector<myType> outputs(size, 0);
		funcRELUPrime3PC(inputs, outputs, size);
		if (PRIMARY)
		{
			funcReconstruct2PC(inputs, size, "inputs");
			funcReconstruct2PC(outputs, size, "outputs");
		}
	}

	if (FOUR_PC)
	{
		vector<smallType> outputs(size, 0);
		funcRELUPrime4PC(inputs, outputs, size);
		if (PRIMARY)
		{
			funcReconstruct2PC(inputs, size, "inputs");
			funcReconstructBit2PC(outputs, size, "outputs");
		}
	}
}


void debugMaxIndex()
{
	size_t rows = 10;
	size_t columns = 4;

	vector<myType> maxIndex(rows, 0);
	if (partyNum == PARTY_A)
		for (size_t i = 0; i < rows; ++i)
			maxIndex[i] = (aes_indep->get8Bits())%columns;

	vector<myType> a(rows*columns);	
	funcMaxIndexMPC(a, maxIndex, rows, columns);

	if (PRIMARY)
	{
		funcReconstruct2PC(maxIndex, maxIndex.size(), "maxIndex");
		
		vector<myType> temp(rows*columns);
		if (partyNum == PARTY_B)
			sendVector<myType>(a, PARTY_A, rows*columns);

		if (partyNum == PARTY_A)
		{
			receiveVector<myType>(temp, PARTY_B, rows*columns);
			addVectors<myType>(temp, a, temp, rows*columns);
		
			cout << "a: " << endl;
			for (size_t i = 0; i < rows; ++i)
			{
				for (int j = 0; j < columns; ++j)
				{
					print_linear(temp[i*columns + j], DEBUG_PRINT);
				}
				cout << endl;
			}
			cout << endl;
		}
	}
}




/******************************** Test ********************************/
void testMatMul(size_t rows, size_t common_dim, size_t columns, size_t iter)
{
	vector<myType> a(rows*common_dim, 1);
	vector<myType> b(common_dim*columns, 1);
	vector<myType> c(rows*columns);

	if (STANDALONE)
	{
		for (int runs = 0; runs < iter; ++runs)
		{
			matrixMultEigen(a, b, c, rows, common_dim, columns, 0, 0);
			dividePlainSA(c, (1 << FLOAT_PRECISION));
		}
	}

	if (MPC)
	{
		for (int runs = 0; runs < iter; ++runs)
			funcMatMulMPC(a, b, c, rows, common_dim, columns, 0, 0);
	}
}


void testConvolution(size_t iw, size_t ih, size_t fw, size_t fh, size_t C, size_t D, size_t iter)
{
	size_t sx = 1, sy = 1, B = MINI_BATCH_SIZE;
	vector<myType> w(fw*fh*C*D, 0);
	vector<myType> act(iw*ih*C*B, 0);
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t size_rw = fw*fh*C*D;
	size_t rows_rw = fw*fh*C;
	size_t columns_rw = D;


	for (int runs = 0; runs < iter; ++runs)
	{
		//Reshape weights
		vector<myType> reshapedWeights(size_rw, 0);
		for (int i = 0; i < size_rw; ++i)
			reshapedWeights[(i%rows_rw)*columns_rw + (i/rows_rw)] = w[i];

		//reshape activations
		size_t size_convo = (p_range*q_range*B) * (fw*fh*C); 
		vector<myType> convShaped(size_convo, 0);
		convolutionReshape(act, convShaped, iw, ih, C, B, fw, fh, 1, 1);


		//Convolution multiplication
		vector<myType> convOutput(p_range*q_range*B*D, 0);
		if (STANDALONE)
		{
			matrixMultEigen(convShaped, reshapedWeights, convOutput, 
						(p_range*q_range*B), (fw*fh*C), D, 0, 0);
			dividePlainSA(convOutput, (1 << FLOAT_PRECISION));
		}

		if (MPC)
		{
			funcMatMulMPC(convShaped, reshapedWeights, convOutput, 
						(p_range*q_range*B), (fw*fh*C), D, 0, 0);
		}
	}
}


void testRelu(size_t r, size_t c, size_t iter)
{
	vector<myType> a(r*c, 1);
	vector<smallType> reluPrimeSmall(r*c, 1);
	vector<myType> reluPrimeLarge(r*c, 1);
	vector<myType> b(r*c, 0);

	for (int runs = 0; runs < iter; ++runs)
	{
		if (STANDALONE)
			for (size_t i = 0; i < r*c; ++i)
				b[i] = a[i] * reluPrimeSmall[i];

		if (FOUR_PC)
			funcSelectShares4PC(a, reluPrimeSmall, b, r*c);

		if (THREE_PC)
			funcSelectShares3PC(a, reluPrimeLarge, b, r*c);
	}
}


void testReluPrime(size_t r, size_t c, size_t iter)
{
	vector<myType> a(r*c, 1);
	vector<myType> b(r*c, 0);
	vector<smallType> d(r*c, 0);

	for (int runs = 0; runs < iter; ++runs)
	{
		if (STANDALONE)
			for (size_t i = 0; i < r*c; ++i)
				b[i] = (a[i] < LARGEST_NEG ? 1:0);

		if (THREE_PC)
			funcRELUPrime3PC(a, b, r*c);

		if (FOUR_PC)
			funcRELUPrime4PC(a, d, r*c);
	}
}


void testMaxPool(size_t p_range, size_t q_range, size_t px, size_t py, size_t D, size_t iter)
{
	size_t B = MINI_BATCH_SIZE;
	size_t size_x = p_range*q_range*D*B;

	vector<myType> y(size_x, 0);
	vector<myType> maxPoolShaped(size_x, 0);
	vector<myType> act(size_x/(px*py), 0);
	vector<myType> maxIndex(size_x/(px*py), 0); 

	for (size_t i = 0; i < iter; ++i)
	{
		maxPoolReshape(y, maxPoolShaped, p_range, q_range, D, B, py, px, py, px);

		if (STANDALONE)
		{
			size_t size = (size_x/(px*py))*(px*py);
			vector<myType> diff(size);

			for (size_t i = 0; i < (size_x/(px*py)); ++i)
			{
				act[i] = maxPoolShaped[i*(px*py)];
				maxIndex[i] = 0;
			}

			for (size_t i = 1; i < (px*py); ++i)
				for (size_t j = 0; j < (size_x/(px*py)); ++j)
				{
					if (maxPoolShaped[j*(px*py) + i] > act[j])
					{
						act[j] = maxPoolShaped[j*(px*py) + i];
						maxIndex[j] = i;
					}
				}
		}
		
		if (MPC)
			funcMaxMPC(maxPoolShaped, act, maxIndex, size_x/(px*py), px*py);
	}
}

void testMaxPoolDerivative(size_t p_range, size_t q_range, size_t px, size_t py, size_t D, size_t iter)
{
	size_t B = MINI_BATCH_SIZE;
	size_t alpha_range = p_range/py;
	size_t beta_range = q_range/px;
	size_t size_y = (p_range*q_range*D*B);
	vector<myType> deltaMaxPool(size_y, 0);
	vector<myType> deltas(size_y/(px*py), 0);
	vector<myType> maxIndex(size_y/(px*py), 0);

	size_t size_delta = alpha_range*beta_range*D*B;
	vector<myType> thatMatrixTemp(size_y, 0), thatMatrix(size_y, 0);


	for (size_t i = 0; i < iter; ++i)
	{
		if (STANDALONE)
			for (size_t i = 0; i < size_delta; ++i)
				thatMatrixTemp[i*px*py + maxIndex[i]] = 1;

		if (MPC)
			funcMaxIndexMPC(thatMatrixTemp, maxIndex, size_delta, px*py);
		

		//Reshape thatMatrix
		size_t repeat_size = D*B;
		size_t alpha_offset, beta_offset, alpha, beta;
		for (size_t r = 0; r < repeat_size; ++r)
		{
			size_t size_temp = p_range*q_range;
			for (size_t i = 0; i < size_temp; ++i)
			{
				alpha = (i/(px*py*beta_range));
				beta = (i/(px*py)) % beta_range;
				alpha_offset = (i%(px*py))/px;
				beta_offset = (i%py);
				thatMatrix[((py*alpha + alpha_offset)*q_range) + 
						   (px*beta + beta_offset) + r*size_temp] 
				= thatMatrixTemp[r*size_temp + i];
			}
		}

		//Replicate delta martix appropriately
		vector<myType> largerDelta(size_y, 0);
		size_t index_larger, index_smaller;
		for (size_t r = 0; r < repeat_size; ++r)
		{
			size_t size_temp = p_range*q_range;
			for (size_t i = 0; i < size_temp; ++i)
			{
				index_smaller = r*size_temp/(px*py) + (i/(q_range*py))*beta_range + ((i%q_range)/px);
				index_larger = r*size_temp + (i/q_range)*q_range + (i%q_range);
				largerDelta[index_larger] = deltas[index_smaller];
			}
		}

		if (STANDALONE)
			for (size_t i = 0; i < size_y; ++i)
				deltaMaxPool[i] = largerDelta[i] * thatMatrix[i]; 

		if (MPC)
			funcDotProductMPC(largerDelta, thatMatrix, deltaMaxPool, size_y);
	}
}


