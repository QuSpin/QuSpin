


class A{
	public:
		A(){}
		~A(){}
};

class B: public A{
	public:
		B(){}
		~B(){}
};

class C: public A{
	public:
		C(){}
		~C(){}
};


class D: public A{
	public:
		D(){}
		~D(){}
};


int main(int argc, char const *argv[])
{
	A * obj = new D();
	return 0;
}
